#!/usr/bin/env python3
"""
SIR Projekat — Faza 2
Self-Refine petlja za prevod sa Skill Persistence
v3 — system prompt via /api/chat, langdetect, preprocesor za stage directions
Balsam server | Flavio & Claude | Mart 2026
"""

import json
import subprocess
import requests
import re
import os
import time
from datetime import datetime

# ── Konfiguracija ──────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434"
TRANSLATE_MODEL = "balsam:latest"
CRITIQUE_MODEL  = "balsam:latest"
EMBED_MODEL     = "llama3.2:3b"
DB_USER         = "pgu"
DB_HOST         = "127.0.0.1"
DB_NAME         = "balsam"
DB_PASS         = "Pgu1234.1234"
PROCEDURE_ID    = 2
MAX_ITERATIONS  = 3
CONV_THRESHOLD  = 0.02

LANG_NAMES = {
    "srp": "Serbian",
    "hrv": "Croatian",
    "bos": "Bosnian"
}

# Skup karaktera koji su tipični za ciljne jezike
LANG_CHARS = {
    "srp": set("abcdefghijklmnoprstuvzšđčćžABCDEFGHIJKLMNOPRSTUVZŠĐČĆŽ аАбБвВгГдДеЕжЖзЗиИјЈкКлЛмМнНоОпПрРсСтТуУфФхХцЦчЧшШ"),
    "hrv": set("abcdefghijklmnoprstuvzšđčćžABCDEFGHIJKLMNOPRSTUVZŠĐČĆŽ"),
    "bos": set("abcdefghijklmnoprstuvzšđčćžABCDEFGHIJKLMNOPRSTUVZŠĐČĆŽ"),
}

# ── Timing helper ──────────────────────────────────────────────────────────────
def timer():
    return time.time()

def elapsed(start):
    return f"{time.time() - start:.1f}s"

# ── Preprocesor ────────────────────────────────────────────────────────────────
def preprocess(text):
    """
    NOVO v3: čisti stage directions i artefakte prije slanja modelu.
    [sighs], [laughter], <i>tekst</i>, itd.
    Vraća (očišćen tekst, lista uklonjenih artefakata).
    """
    artifacts = re.findall(r'\[.*?\]', text)
    clean = re.sub(r'\[.*?\]', '', text)
    clean = re.sub(r'<[^>]+>', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean, artifacts

# ── Detekcija jezika izlaza ────────────────────────────────────────────────────
def is_target_lang(text, lang):
    """
    NOVO v3: provjera da li izlaz sadrži dovoljno karaktera ciljnog jezika.
    Jednostavna heuristika — ne zahtijeva vanjsku biblioteku.
    """
    if not text or len(text) < 3:
        return False
    lang_set = LANG_CHARS.get(lang, set())
    matches = sum(1 for c in text if c in lang_set)
    ratio = matches / max(len(text), 1)
    # Za srp/hrv/bos očekujemo bar 30% karaktera iz skupa
    return ratio > 0.30

# ── DB helper ──────────────────────────────────────────────────────────────────
def psql(query):
    cmd = ["docker", "exec", "pgdb",
           "psql", "-h", DB_HOST, "-U", DB_USER, "-d", DB_NAME, "-t", "-c", query]
    env = {**os.environ, "PGPASSWORD": DB_PASS}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.stdout.strip()

# ── Ollama /api/chat helpers ───────────────────────────────────────────────────
def chat(system_msg, user_msg, model, temperature=0.3):
    """
    NOVO v3: koristi /api/chat umjesto /api/generate.
    System prompt je strukturalno odvojen od user sadržaja.
    """
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg}
        ],
        "stream": False,
        "options": {"temperature": temperature}
    }, timeout=180)
    return resp.json().get("message", {}).get("content", "").strip()

def translate(text, target_lang, model=TRANSLATE_MODEL, critique=""):
    system = (
        f"You are a professional translator. "
        f"Your only task is to translate text into {target_lang}. "
        f"Return ONLY the translated text. "
        f"Do not explain, comment, or repeat the original. "
        f"Do not include any meta-instructions in your output."
    )
    user = f"Translate into {target_lang}:\n{text}"
    if critique:
        user += f"\n\nApply this feedback silently:\n{critique}"

    t = timer()
    result = chat(system, user, model, temperature=0.3)
    print(f"    [translate {elapsed(t)}]")
    return result

def self_critique(original, translation, reference, similarity, target_lang):
    system = (
        "You are a translation quality evaluator. "
        "Respond in English. Focus only on translation quality issues. "
        "Be concise — list 1-2 specific problems and suggest improvements."
    )
    user = (
        f"Original (English): {original}\n"
        f"Translation ({target_lang}): {translation}\n"
        f"Reference ({target_lang}): {reference}\n"
        f"Quality score: {similarity:.3f}/1.0\n\n"
        f"What are the main issues with this translation?"
    )
    t = timer()
    result = chat(system, user, CRITIQUE_MODEL, temperature=0.2)
    print(f"    [critique {elapsed(t)}]")
    return result

def get_embedding(text, model=EMBED_MODEL):
    t = timer()
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
        "model": model, "prompt": text
    }, timeout=30)
    print(f"    [embed {elapsed(t)}]")
    return resp.json().get("embedding", [])

def cosine_similarity(v1, v2):
    if not v1 or not v2:
        return 0.0
    dot = sum(a*b for a,b in zip(v1,v2))
    n1  = sum(a*a for a in v1)**0.5
    n2  = sum(b*b for b in v2)**0.5
    return dot/(n1*n2) if n1 and n2 else 0.0

# ── DB upis ────────────────────────────────────────────────────────────────────
def save_trajectory(lang, direction, original, iterations, final_sim, baseline_sim, steps_log, success):
    delta = final_sim - baseline_sim
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
    dj = json.dumps(details or {}).replace("'", "''")
    psql(f"""INSERT INTO sir_metadata (session_id, lang, metric, value, details)
             VALUES ('{session_id}', '{lang}', '{metric}', {value}, '{dj}'::jsonb);""")

# ── Self-Refine petlja ─────────────────────────────────────────────────────────
def self_refine(original, reference, lang, direction, session_id):
    t_total = timer()
    target_lang = LANG_NAMES.get(lang, lang)

    # Preprocesor
    clean_original, artifacts = preprocess(original)
    if artifacts:
        print(f"  [preproc] Uklonjeni artefakti: {artifacts}")

    print(f"\n{'─'*60}")
    print(f"  {lang} ({target_lang}) | {direction}")
    print(f"  Orig:  {original[:70]}")
    if clean_original != original:
        print(f"  Clean: {clean_original[:70]}")
    print(f"  Ref:   {reference[:70]}")

    ref_emb   = get_embedding(reference)
    steps_log = []
    critique  = ""
    prev_sim  = 0.0

    for i in range(1, MAX_ITERATIONS + 1):
        t_iter = timer()
        print(f"\n  [Iter {i}]")

        translation = translate(clean_original, target_lang, critique=critique)
        print(f"  → {translation[:80]}")

        # NOVO v3: provjera jezika izlaza
        lang_ok = is_target_lang(translation, lang)
        if not lang_ok:
            print(f"  ⚠ Izlaz nije na {target_lang} — označavam kao neuspjeh")

        t_emb = get_embedding(translation)
        sim   = cosine_similarity(t_emb, ref_emb)
        delta = sim - prev_sim
        print(f"  Similarity: {sim:.4f}  Δ{delta:+.4f}  lang_ok={lang_ok}  [{elapsed(t_iter)}]")

        step = {
            "iteration": i,
            "translation": translation,
            "similarity": round(sim, 6),
            "delta": round(delta, 6),
            "lang_ok": lang_ok
        }

        if i < MAX_ITERATIONS:
            if abs(delta) < CONV_THRESHOLD and i > 1:
                print(f"  Konvergencija — zaustavljam.")
                steps_log.append(step)
                break
            critique = self_critique(clean_original, translation, reference, sim, target_lang)
            print(f"  Kritika: {critique[:100]}")
            step["critique"] = critique

        steps_log.append(step)
        prev_sim = sim

    baseline_sim = steps_log[0]["similarity"]
    # NOVO best_so_far: uzimamo iteraciju s najvišom similarity, ne zadnju
    best_step    = max(steps_log, key=lambda s: s["similarity"])
    final_sim    = best_step["similarity"]
    final_lang_ok = best_step.get("lang_ok", True)
    if best_step != steps_log[-1]:
        print(f"  [best_so_far] Iter {best_step['iteration']} ({final_sim:.4f}) > Zadnja ({steps_log[-1]['similarity']:.4f})")
    success = (final_sim > baseline_sim) and final_lang_ok

    t = timer()
    traj_id = save_trajectory(lang, direction, original, len(steps_log),
                               final_sim, baseline_sim, steps_log, success)
    print(f"  [db write {elapsed(t)}]")
    print(f"\n  Baseline {baseline_sim:.4f} → Final {final_sim:.4f}  ({final_sim-baseline_sim:+.4f})  lang_ok={final_lang_ok}")
    print(f"  Trajektorija ID: {traj_id}  |  Ukupno: {elapsed(t_total)}")

    return {
        "traj_id": traj_id, "lang": lang, "direction": direction,
        "iterations": len(steps_log), "baseline": baseline_sim,
        "final": final_sim, "delta": final_sim - baseline_sim,
        "success": success, "lang_ok": final_lang_ok
    }

# ── Uzorci iz baze ─────────────────────────────────────────────────────────────
def get_samples(lang, n=2):
    rows = psql(f"""SELECT local_text, eng_text FROM sentence_pairs_v2
                    WHERE lang='{lang}' ORDER BY RANDOM() LIMIT {n};""")
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
    t_session = timer()
    session_id = f"sir_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nSIR Self-Refine v3 | Sesija: {session_id}")
    print(f"Model: {TRANSLATE_MODEL} | Kritika: {CRITIQUE_MODEL} | Emb: {EMBED_MODEL}")
    print(f"Max iteracija: {MAX_ITERATIONS} | Konvergencija: {CONV_THRESHOLD}")
    print(f"Endpoint: /api/chat (system prompt odvojen od sadrzaja)")

    results = []

    for lang in ["srp", "hrv", "bos"]:
        samples = get_samples(lang, n=2)
        if not samples:
            print(f"  Nema uzoraka za {lang}")
            continue
        for local_text, eng_text in samples:
            r = self_refine(eng_text, local_text, lang, "eng2local", session_id)
            results.append(r)

    for lang in ["srp", "hrv", "bos"]:
        lr = [r for r in results if r["lang"] == lang]
        if lr:
            avg_delta = sum(r["delta"] for r in lr) / len(lr)
            avg_final = sum(r["final"] for r in lr) / len(lr)
            save_metadata(session_id, lang, "avg_delta_similarity", avg_delta,
                         {"n": len(lr), "avg_final": avg_final})

    print(f"\n{'='*60}")
    print(f"  SESIJA: {session_id}")
    print(f"  Trajektorija: {len(results)}")
    ok = sum(1 for r in results if r["success"])
    lang_ok_count = sum(1 for r in results if r["lang_ok"])
    print(f"  Uspjesnih (sim + lang_ok): {ok}/{len(results)}")
    print(f"  Lang check OK: {lang_ok_count}/{len(results)}")
    if results:
        avg = sum(r["delta"] for r in results) / len(results)
        print(f"  Prosjecno delta similarity: {avg:+.4f}")
    print(f"  Ukupno vrijeme sesije: {elapsed(t_session)}")
    print(f"{'='*60}")
