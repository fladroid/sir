#!/usr/bin/env python3
"""
SIR Projekat — Prsten 1: Perceptivni Agent v5
Ključna novost: pre-scoring uzoraka
  - Kandidati se embeduju i scored PRIJE Self-Refine
  - Samo uzorci s baseline 0.30-0.70 idu u petlju
  - Garantuje da agent uvijek radi u optimalnoj zoni
  - stream:False u svim Ollama pozivima
  - best_so_far logika (čuva se max similarity, ne final)
Flavio & Claude | Mart 2026
"""

import json, subprocess, requests, re, time, sys, math
from datetime import datetime

OLLAMA_URL          = "http://localhost:11434"
PLAN_MODEL          = "balsam:latest"
EMBED_MODEL         = "llama3.2:3b"
DB_USER, DB_HOST, DB_NAME = "pgu", "127.0.0.1", "balsam"
PROCEDURE_ID        = 5
BASELINE_LOW        = 0.30
BASELINE_HIGH       = 0.70
WORD_MIN, WORD_MAX  = 5, 20
MAX_N               = 3
CANDIDATES_POOL     = 20   # uzorci za pre-scoring po jeziku

def timer(): return time.time()
def elapsed(s): return f"{time.time()-s:.1f}s"

def psql(q):
    cmd = ["docker","exec","pgdb","psql","-h",DB_HOST,"-U",DB_USER,"-d",DB_NAME,"-t","-c",q]
    return subprocess.run(cmd, capture_output=True, text=True).stdout

def llm_chat(sys_msg, usr_msg, timeout=90):
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=timeout, json={
        "model": PLAN_MODEL, "stream": False,
        "messages": [{"role":"system","content":sys_msg},{"role":"user","content":usr_msg}]
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

def get_embedding(text, timeout=30):
    r = requests.post(f"{OLLAMA_URL}/api/embed", timeout=timeout, json={
        "model": EMBED_MODEL, "input": text
    })
    r.raise_for_status()
    return r.json()["embeddings"][0]

def cosine(a, b):
    dot   = sum(x*y for x,y in zip(a,b))
    normA = math.sqrt(sum(x**2 for x in a))
    normB = math.sqrt(sum(x**2 for x in b))
    return dot / (normA * normB) if normA * normB > 0 else 0.0

def parse_json_robust(text):
    for attempt in [text, ' '.join(text.split())]:
        try: return json.loads(attempt)
        except: pass
        m = re.search(r'\{[^{}]*\}', attempt)
        if m:
            try: return json.loads(m.group())
            except: pass
    return None

def save_insight(sid, typ, content, meta=None):
    c = content.replace("'","''")
    m = json.dumps(meta or {}).replace("'","''")
    psql(f"INSERT INTO sir_insights(session_id,insight_type,content,meta) VALUES('{sid}','{typ}','{c}','{m}');")
    print(f"  [INSIGHT] {typ}: {content[:90]}")

# ── Percepcija ─────────────────────────────────────────────────────────────────
def perceive(sid):
    print("\n── PERCEPCIJA ─────────────────────────────────────────────────────")
    stats = psql(f"""
        SELECT lang,
               COUNT(*) as n,
               COUNT(*) FILTER (WHERE (final_similarity-delta_similarity)
                                BETWEEN {BASELINE_LOW} AND {BASELINE_HIGH}) as u_zoni,
               ROUND(AVG(delta_similarity) FILTER (
                   WHERE (final_similarity-delta_similarity) BETWEEN {BASELINE_LOW} AND {BASELINE_HIGH}
               )::numeric,4) as delta_zoni,
               ROUND(AVG(delta_similarity)::numeric,4) as delta_ukupno,
               COUNT(*) FILTER (WHERE success=true) as ok
        FROM sir_trajectories
        GROUP BY lang ORDER BY delta_zoni DESC NULLS LAST;
    """)
    print("lang | n | u_zoni | delta_zoni | delta_ukupno | ok")
    print(stats)
    return stats

# ── Planiranje ─────────────────────────────────────────────────────────────────
def plan(stats, sid):
    print("\n── PLANIRANJE v5 ──────────────────────────────────────────────────")
    system = (
        "Orchestrator for Self-Refine translation system.\n"
        "RULE: prioritize language with HIGHEST POSITIVE delta_zoni (first row = best).\n"
        "Respond ONE LINE JSON only:\n"
        '{"priority_lang":"srp","reason":"short","n_samples":2,"strategy":"focus_weak"}\n'
        "n_samples:1-3. strategy:focus_weak|balanced|test_strong."
    )
    user = f"Stats(lang|n|u_zoni|delta_zoni|delta_ukupno|ok):\n{stats.strip()}\nChoose. One JSON line."
    t = timer()
    raw = ""
    try:
        raw = llm_chat(system, user)
        print(f"  LLM ({elapsed(t)}): {raw[:100]}")
        d = parse_json_robust(raw)
        if not d: raise ValueError("parse fail")
        lang = d.get("priority_lang","srp")
        if lang not in ("srp","hrv","bos"): lang = "srp"
        d["priority_lang"] = lang
        d["n_samples"] = max(1, min(int(d.get("n_samples",2)), MAX_N))
        d["strategy"] = str(d.get("strategy","balanced")).split("|")[0].strip()
    except Exception as e:
        print(f"  [WARN] fallback srp ({e})")
        d = {"priority_lang":"srp","reason":f"fallback:{e}","n_samples":2,"strategy":"balanced"}
    print(f"  → {d['priority_lang']} | {d['strategy']} | n={d['n_samples']} | {d.get('reason','')}")
    save_insight(sid, "planning_v5", json.dumps(d), {"raw": raw[:200]})
    return d

# ── Pre-scoring uzoraka ────────────────────────────────────────────────────────
def prescored_sample(lang, n, sid):
    """
    Uzmi CANDIDATES_POOL kandidata, quick-score svakog (embed eng + embed local),
    vrati n koji su u optimalnoj zoni (0.30-0.70).
    """
    print(f"\n── PRE-SCORING ({lang.upper()}, tražim {n} u zoni) ─────────────────")

    raw = psql(f"""
        SELECT local_text, eng_text
        FROM sentence_pairs_v2
        WHERE lang='{lang}'
          AND array_length(string_to_array(eng_text,' '),1) BETWEEN {WORD_MIN} AND {WORD_MAX}
          AND id NOT IN (
              SELECT sp.id FROM sentence_pairs_v2 sp
              JOIN sir_trajectories st ON st.original_text = sp.eng_text
              WHERE sp.lang='{lang}')
        ORDER BY RANDOM() LIMIT {CANDIDATES_POOL};
    """)

    candidates = []
    for line in raw.split('\n'):
        line = line.strip()
        if '|' in line:
            p = [x.strip() for x in line.split('|')]
            if len(p) >= 2 and p[0] and p[1]:
                candidates.append((p[0], p[1]))

    print(f"  Pool: {len(candidates)} kandidata → scoring...")
    scored = []
    for local, eng in candidates:
        try:
            t = timer()
            emb_eng   = get_embedding(eng)
            emb_local = get_embedding(local)
            sim = cosine(emb_eng, emb_local)
            scored.append((local, eng, sim))
            status = "✓ zona" if BASELINE_LOW <= sim <= BASELINE_HIGH else f"  {sim:.3f}"
            print(f"  {status} | {eng[:50]}")
        except Exception as e:
            print(f"  [skip] embed error: {e}")

    in_zone = [(l,e,s) for l,e,s in scored if BASELINE_LOW <= s <= BASELINE_HIGH]
    print(f"  U zoni: {len(in_zone)}/{len(scored)}")

    selected = in_zone[:n]
    if len(selected) < n:
        print(f"  [WARN] Samo {len(selected)}/{n} uzoraka u zoni za {lang}")

    return [(l, e, s) for l, e, s in selected]

# ── Self-Refine ────────────────────────────────────────────────────────────────
def run_refine(samples, lang, sid):
    sys.path.insert(0, '/home/balsam/sir/faza1')
    import self_refine_v3 as sr
    sr.PROCEDURE_ID = PROCEDURE_ID
    results = []
    for item in samples:
        # v3.1: sample je (local, eng, pre_score_sim) ili (local, eng)
        if len(item) == 3:
            local, eng, pre_score_sim = item
        else:
            local, eng = item
            pre_score_sim = None
        print(f"\n  → SR: {eng[:60]}... (pre_score={pre_score_sim:.3f})" if pre_score_sim else f"\n  → SR: {eng[:60]}...")
        results.append(sr.self_refine(eng, local, lang, "eng2local", sid, pre_score_sim=pre_score_sim))
    return results

# ── Opservacija ────────────────────────────────────────────────────────────────
def observe(results, decision, sid):
    print("\n── OPSERVACIJA ────────────────────────────────────────────────────")
    if not results: return
    n   = len(results)
    ok  = sum(1 for r in results if r.get("success"))
    lok = sum(1 for r in results if r.get("lang_ok"))
    avg_d = sum(r.get("delta",0) for r in results)/n
    avg_f = sum(r.get("final",0) for r in results)/n
    print(f"  {ok}/{n} uspješnih | lang_ok {lok}/{n} | avg_delta {avg_d:+.4f} | avg_final {avg_f:.4f}")

    summary = {"n":n,"ok":ok,"lok":lok,"avg_delta":round(avg_d,4),
               "avg_final":round(avg_f,4),"lang":decision["priority_lang"]}
    per = [{"d":round(r.get("delta",0),3),"ok":r.get("success")} for r in results]

    try:
        insight = llm_chat(
            "State in 1-2 sentences the key pattern in this Self-Refine session. Plain text, no JSON.",
            f"Summary:{json.dumps(summary)} Per-result:{json.dumps(per)}",
            timeout=60
        )
        print(f"\n  [REFLEKSIJA] {insight}")
        save_insight(sid, "reflection_v5", insight, summary)
    except Exception as e:
        save_insight(sid, "reflection_v5", f"error:{e}", summary)

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0  = timer()
    sid = f"agent5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"  SIR Perceptivni Agent v5 | {sid}")
    print(f"  Procedure: {PROCEDURE_ID} | Pre-scoring pool: {CANDIDATES_POOL}")
    print(f"  Ciljna zona: baseline {BASELINE_LOW}–{BASELINE_HIGH}")
    print(f"{'='*60}")

    stats  = perceive(sid)
    dec    = plan(stats, sid)
    pri    = dec["priority_lang"]
    n      = dec["n_samples"]

    # Prioritetni jezik: pre-scoring, ostatak: 1 uzorak bez scoring
    all_samples = {pri: prescored_sample(pri, n, sid)}
    for lang in ["srp","hrv","bos"]:
        if lang != pri:
            # Za sekundarne jezike: brzo uzorkovanje bez scoring (1 uzorak)
            raw = psql(f"""SELECT local_text, eng_text FROM sentence_pairs_v2
                WHERE lang='{lang}'
                  AND array_length(string_to_array(eng_text,' '),1) BETWEEN {WORD_MIN} AND {WORD_MAX}
                  AND id NOT IN (SELECT sp.id FROM sentence_pairs_v2 sp
                      JOIN sir_trajectories st ON st.original_text=sp.eng_text WHERE sp.lang='{lang}')
                ORDER BY RANDOM() LIMIT 1;""")
            s = []
            for line in raw.split('\n'):
                line = line.strip()
                if '|' in line:
                    p = [x.strip() for x in line.split('|')]
                    if len(p)>=2 and p[0] and p[1]: s.append((p[0],p[1]))
            all_samples[lang] = s
            if s: print(f"\n── UZORAK ({lang.upper()}, bez scoring) ─────────────────────────\n  {s[0][1][:65]}")

    all_results = []
    for lang, samp in all_samples.items():
        if samp:
            all_results.extend(run_refine(samp, lang, sid))

    observe(all_results, dec, sid)

    ok  = sum(1 for r in all_results if r.get("success"))
    avg = sum(r.get("delta",0) for r in all_results)/len(all_results) if all_results else 0
    print(f"\n{'='*60}")
    print(f"  {sid} | {len(all_results)} trajektorija | {ok}/{len(all_results)} OK")
    print(f"  Avg delta: {avg:+.4f} | Ukupno: {elapsed(t0)}")
    print(f"{'='*60}")
