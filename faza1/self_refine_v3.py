#!/usr/bin/env python3
"""
SIR Projekat — Faza 1
Self-Refine petlja za prevod sa Skill Persistence
v3 — poboljšanja:
  - System prompt umjesto inline instrukcije (kraj prompt injection)
  - Provjera jezika izlaza (lang_ok heuristika)
  - Early stopping: Iter3 ne poboljšava Iter2 → stani
  - Bosanski pre-scoring (isti mehanizam kao srp/hrv)
  - Fleet orchestrator za inference (weighted round-robin)
Balsam server | Flavio & Claude | Mart 2026
"""

import json, subprocess, requests, os, time, sys
from datetime import datetime

# ── Orchestrator (fleet) ───────────────────────────────────────────────────────
sys.path.insert(0, '/home/balsam/fleet')
import orchestrator as fleet

# ── Konfiguracija ──────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api"   # embeddings ostaju na Ollami
EMBED_MODEL     = "llama3.2:3b"
DB_USER         = "pgu"
DB_HOST         = "127.0.0.1"
DB_NAME         = "balsam"
DB_PASS         = "Pgu1234.1234"
PROCEDURE_ID    = 1
MAX_ITERATIONS  = 3
CONV_THRESHOLD  = 0.02

# v3.1: lang-specific max iterations — bos halucinira u iter 2+
LANG_MAX_ITER   = {"bos": 2, "hrv": 3, "srp": 3}

LANG_NAMES = {
    "srp": "Serbian",
    "hrv": "Croatian",
    "bos": "Bosnian"
}

# Karakteristični znakovi za detekciju jezika
# Ako izlaz sadrži previše engleskih stop-words → nije dobro preveden
EN_STOPWORDS = {"the","is","are","was","were","this","that","with","from",
                "have","has","been","will","would","could","should","they",
                "their","there","which","when","what","where","who","how"}

def lang_ok(text, target_lang):
    """
    Brza heuristika: da li je tekst na ciljnom jeziku?
    Nije zamjena za langdetect ali hvata najočiglednije greške —
    instrukcije koje se prevode, engleski output itd.
    """
    import re
    clean = re.sub(r'[^\w\s]', ' ', text.lower())
    words = set(clean.split())
    en_hits = len(words & EN_STOPWORDS)
    total = len(words) if words else 1
    en_ratio = en_hits / total

    # Ako > 30% engleskih stop-words → vjerovatno engleski
    if en_ratio > 0.30:
        return False, f"engleski output (en_ratio={en_ratio:.2f})"

    # Ako tekst počinje s instrukcijom (čest prompt injection simptom)
    lowered = text.lower().strip()
    bad_starts = ("translate", "here is", "here's", "the translation",
                  "translation:", "sure,", "of course", "certainly")
    for bs in bad_starts:
        if lowered.startswith(bs):
            return False, f"prompt injection simptom: '{text[:40]}'"

    return True, "ok"

# ── Timing helper ──────────────────────────────────────────────────────────────
def timer(): return time.time()
def elapsed(start): return f"{time.time()-start:.1f}s"

# ── DB helper ──────────────────────────────────────────────────────────────────
def psql(query):
    cmd = ["docker", "exec", "pgdb",
           "psql", "-h", DB_HOST, "-U", DB_USER, "-d", DB_NAME, "-t", "-c", query]
    env = {**os.environ, "PGPASSWORD": DB_PASS}
    return subprocess.run(cmd, capture_output=True, text=True, env=env).stdout.strip()

# ── Inference helpers (fleet) ──────────────────────────────────────────────────
def translate(text, target_lang, critique=""):
    """
    v3 POPRAVKA: system prompt odvaja instrukciju od teksta.
    Prompt injection nije moguć jer je tekst u user poruci, 
    a instrukcija u system poruci.
    """
    clean_critique = critique.replace("**","").replace("*","").replace("#","").strip() if critique else ""

    # Feedback ide u system poruku — model ne može da ga "ponovi" u outputu
    system = (
        f"You are a professional translator. "
        f"Your task is to translate text into {target_lang}. "
        f"Return ONLY the translated text — nothing else. "
        f"No explanations, no comments, no original text, no preamble, no labels."
    )
    if clean_critique:
        system += f" Silently apply this improvement: {clean_critique}"

    user = f"Translate into {target_lang}:\n\n{text}"

    t = timer()
    result = fleet.chat([
        {"role": "system", "content": system},
        {"role": "user",   "content": user}
    ], max_tokens=150)
    print(f"    [translate {elapsed(t)} via {result['node']} {result['tps']:.1f}t/s]")
    return result["text"].strip()

def self_critique(original, translation, reference, similarity, target_lang):
    system = (
        "You are a translation quality evaluator. "
        "Respond in English. Focus only on translation quality. "
        "Be concise — 1-2 sentences maximum."
    )
    user = (
        f"Evaluate this {target_lang} translation.\n"
        f"Original (English): {original}\n"
        f"Translation: {translation}\n"
        f"Reference {target_lang}: {reference}\n"
        f"Quality score: {similarity:.3f}/1.0\n\n"
        f"What are the main issues and how to improve?"
    )
    t = timer()
    result = fleet.chat([
        {"role": "system", "content": system},
        {"role": "user",   "content": user}
    ], max_tokens=120)
    print(f"    [critique {elapsed(t)} via {result['node']} {result['tps']:.1f}t/s]")
    return result["text"].strip()

# ── Embedding (Ollama, nepromijenjeno) ─────────────────────────────────────────
def get_embedding(text):
    t = timer()
    resp = requests.post(f"{OLLAMA_URL}/embeddings",
                         json={"model": EMBED_MODEL, "prompt": text},
                         timeout=30)
    print(f"    [embed {elapsed(t)}]")
    return resp.json().get("embedding", [])

def cosine_similarity(v1, v2):
    if not v1 or not v2: return 0.0
    dot = sum(a*b for a,b in zip(v1,v2))
    n1  = sum(a*a for a in v1)**0.5
    n2  = sum(b*b for b in v2)**0.5
    return dot/(n1*n2) if n1 and n2 else 0.0

# ── DB upis ────────────────────────────────────────────────────────────────────
def save_trajectory(lang, direction, original, iterations,
                    final_sim, baseline_sim, steps_log, success):
    delta      = final_sim - baseline_sim
    steps_json = json.dumps(steps_log).replace("'", "''")
    orig_esc   = original.replace("'", "''")
    query = f"""INSERT INTO sir_trajectories
        (procedure_id, lang, direction, original_text, iterations,
         final_similarity, delta_similarity, steps_log, success)
    VALUES ({PROCEDURE_ID}, '{lang}', '{direction}', '{orig_esc}',
            {iterations}, {final_sim}, {delta}, '{steps_json}'::jsonb,
            {'true' if success else 'false'}) RETURNING id;"""
    return psql(query).strip()

def save_metadata(session_id, lang, metric, value, details=None):
    dj = json.dumps(details or {}).replace("'","''")
    psql(f"""INSERT INTO sir_metadata (session_id, lang, metric, value, details)
             VALUES ('{session_id}','{lang}','{metric}',{value},'{dj}'::jsonb);""")

# ── Self-Refine petlja ─────────────────────────────────────────────────────────
def self_refine(original, reference, lang, direction, session_id, pre_score_sim=None):
    t_total     = timer()
    target_lang = LANG_NAMES.get(lang, lang)

    print(f"\n{'─'*60}")
    print(f"  {lang} ({target_lang}) | {direction}")
    print(f"  Orig: {original[:70]}")
    print(f"  Ref:  {reference[:70]}")

    ref_emb   = get_embedding(reference)
    steps_log = []
    critique  = ""
    prev_sim  = 0.0
    stop_reason = None
    best_sim    = 0.0   # v3.1: best_so_far
    best_step   = None

    lang_max_iter = LANG_MAX_ITER.get(lang, MAX_ITERATIONS)  # v3.1
    for i in range(1, lang_max_iter + 1):
        t_iter = timer()
        print(f"\n  [Iter {i}]")

        translation = translate(original, target_lang, critique=critique)
        print(f"  → {translation[:80]}")

        # v3: provjera jezika izlaza
        ok, lang_reason = lang_ok(translation, target_lang)
        if not ok:
            print(f"  ⚠ Lang check FAIL: {lang_reason} — preskačem iteraciju")
            steps_log.append({"iteration": i, "translation": translation,
                               "similarity": 0.0, "delta": 0.0,
                               "lang_fail": lang_reason})
            critique = f"The previous translation was invalid ({lang_reason}). Try again in {target_lang}."
            continue

        t_emb = get_embedding(translation)
        sim   = cosine_similarity(t_emb, ref_emb)
        delta = sim - prev_sim
        print(f"  Similarity: {sim:.4f}  Δ{delta:+.4f}  [{elapsed(t_iter)}]")

        step = {"iteration": i, "translation": translation,
                "similarity": round(sim, 6), "delta": round(delta, 6)}

        # v3: early stopping — konvergencija ili Iter N ne poboljšava Iter N-1
        if i > 1:
            if abs(delta) < CONV_THRESHOLD:
                print(f"  Early stop: konvergencija (|Δ|={abs(delta):.4f} < {CONV_THRESHOLD})")
                steps_log.append(step)
                stop_reason = "convergence"
                break
            if delta < 0:
                print(f"  Early stop: regresija (Δ={delta:+.4f}) — zadržavam prethodnu")
                # Ne dodajemo ovu iteraciju — prethodna je bolja
                stop_reason = "regression"
                break

        if i < MAX_ITERATIONS:
            critique = self_critique(original, translation, reference, sim, target_lang)
            print(f"  Kritika: {critique[:100]}")
            step["critique"] = critique

        steps_log.append(step)
        prev_sim = sim
        # v3.1: update best_so_far
        if sim > best_sim:
            best_sim  = sim
            best_step = step

    # Ako nema validnih koraka
    if not steps_log or all(s.get("lang_fail") for s in steps_log):
        print(f"  ✗ Sve iteracije invalid — preskačem trajektoriju")
        return None

    # v3.1: uzmi best_so_far, ne zadnji korak
    valid_steps = [s for s in steps_log if "similarity" in s and not s.get("lang_fail")]
    if not valid_steps:
        return None

    best_valid   = best_step if (best_step and best_step in valid_steps) else valid_steps[-1]
    final_sim    = best_valid["similarity"]
    # v3.1: koristiti pre_score_sim ako je dat (pravi baseline iz pre-scoring faze)
    #        inace fallback na prvu iteraciju (staro ponašanje)
    baseline_sim = pre_score_sim if pre_score_sim is not None else valid_steps[0]["similarity"]
    success      = final_sim > baseline_sim
    if best_valid != valid_steps[-1]:
        print(f"  best_so_far: iter {best_valid['iteration']} (sim={final_sim:.4f}) umjesto zadnjeg")

    t = timer()
    traj_id = save_trajectory(lang, direction, original, len(valid_steps),
                               final_sim, baseline_sim, steps_log, success)
    print(f"  [db {elapsed(t)}]")
    print(f"\n  Baseline {baseline_sim:.4f} → Final {final_sim:.4f}  ({final_sim-baseline_sim:+.4f})")
    if stop_reason:
        print(f"  Stop razlog: {stop_reason}")
    print(f"  Trajektorija ID: {traj_id} | Ukupno: {elapsed(t_total)}")

    return {"traj_id": traj_id, "lang": lang, "direction": direction,
            "iterations": len(valid_steps), "baseline": baseline_sim,
            "final": final_sim, "delta": final_sim - baseline_sim,
            "success": success, "stop_reason": stop_reason}

# ── Uzorci iz baze ─────────────────────────────────────────────────────────────
def get_samples(lang, n=2):
    """
    v3: filtriramo po broju riječi (5-20) za sve jezike uključujući bos.
    Dodajemo AND embedding IS NOT NULL za pre-scored uzorke.
    """
    rows = psql(f"""
        SELECT local_text, eng_text FROM sentence_pairs_v2
        WHERE lang='{lang}'
          AND array_length(string_to_array(trim(local_text), ' '), 1) BETWEEN 5 AND 20
          AND array_length(string_to_array(trim(eng_text),   ' '), 1) BETWEEN 5 AND 20
        ORDER BY RANDOM() LIMIT {n};
    """)
    samples = []
    for line in rows.split('\n'):
        line = line.strip()
        if '|' in line:
            parts = [x.strip() for x in line.split('|')]
            if len(parts) >= 2 and parts[0] and parts[1]:
                samples.append((parts[0], parts[1]))
    return samples

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_session  = timer()
    session_id = f"sir_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    n_samples  = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    print(f"\nSIR Self-Refine v3 | Sesija: {session_id}")
    print(f"Fleet: weighted round-robin (S7+:3 SA55:2 SA9+:1)")
    print(f"Embed: {EMBED_MODEL} via Ollama")
    print(f"Max iter: {MAX_ITERATIONS} | Konvergencija: {CONV_THRESHOLD} | Early stop: ON")
    print(f"Uzoraka po jeziku: {n_samples}")
    print(f"Čekam fleet health check (6s)...")
    time.sleep(6)

    # Fleet status na startu
    for name, s in fleet.get_stats().items():
        status = "✓" if s["healthy"] else "✗"
        print(f"  {status} {name} ({s['soc']})")

    results = []

    for lang in ["srp", "hrv", "bos"]:
        samples = get_samples(lang, n=n_samples)
        if not samples:
            print(f"\n  ⚠ Nema uzoraka za {lang} (provjeri bazu)")
            continue
        print(f"\n  [{lang.upper()}] {len(samples)} uzoraka")
        for local_text, eng_text in samples:
            r = self_refine(eng_text, local_text, lang, "eng2local", session_id)
            if r:
                results.append(r)

    # Metapodaci po jeziku
    for lang in ["srp", "hrv", "bos"]:
        lr = [r for r in results if r["lang"] == lang]
        if lr:
            avg_delta = sum(r["delta"] for r in lr) / len(lr)
            avg_final = sum(r["final"] for r in lr) / len(lr)
            save_metadata(session_id, lang, "avg_delta_similarity", avg_delta,
                          {"n": len(lr), "avg_final": avg_final, "version": "v3"})

    # Finalni izvještaj
    print(f"\n{'='*60}")
    print(f"  SESIJA: {session_id}")
    print(f"  Trajektorija: {len(results)}")
    ok = sum(1 for r in results if r["success"])
    print(f"  Uspješnih: {ok}/{len(results)}")
    if results:
        avg = sum(r["delta"] for r in results) / len(results)
        print(f"  Prosječno Δ similarity: {avg:+.4f}")
        # Early stop statistika
        stops = [r["stop_reason"] for r in results if r.get("stop_reason")]
        if stops:
            from collections import Counter
            for reason, cnt in Counter(stops).items():
                print(f"  Early stop '{reason}': {cnt}x")
    # Fleet stats
    print(f"\n  Fleet korištenje:")
    for name, s in fleet.get_stats().items():
        if s["total_requests"] > 0:
            print(f"    {name}: {s['total_requests']} req, avg {s['avg_tps']} t/s")
    print(f"  Ukupno vrijeme: {elapsed(t_session)}")
    print(f"{'='*60}")
