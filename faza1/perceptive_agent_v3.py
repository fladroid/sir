#!/usr/bin/env python3
"""
SIR Projekat — Prsten 1: Perceptivni Agent v3
Ispravke vs v2:
  - Percepcija sortira po avg_delta_u_zoni DESC — agent vidi koji jezik STVARNO
    profitira od Self-Refine (pozitivan delta u optimalnoj zoni)
  - Planning system prompt: eksplicitno "HIGHEST positive avg_delta_u_zoni = top priority"
  - Procedure ID: 5 (nova procedura za v3 agentic runove)
  - n_samples default: 2 (konzervativnije)
Flavio & Claude | Mart 2026
"""

import json, subprocess, requests, re, time, sys
from datetime import datetime

OLLAMA_URL          = "http://localhost:11434"
PLAN_MODEL          = "balsam:latest"
DB_USER, DB_HOST, DB_NAME = "pgu", "127.0.0.1", "balsam"
PROCEDURE_ID        = 5
BASELINE_LOW        = 0.30
BASELINE_HIGH       = 0.70
WORD_MIN, WORD_MAX  = 5, 20
MAX_N               = 3

def timer(): return time.time()
def elapsed(s): return f"{time.time()-s:.1f}s"

def psql(q):
    cmd = ["docker","exec","pgdb","psql","-h",DB_HOST,"-U",DB_USER,"-d",DB_NAME,"-t","-c",q]
    return subprocess.run(cmd, capture_output=True, text=True).stdout

def llm_chat(sys_msg, usr_msg, timeout=90):
    r = requests.post(f"{OLLAMA_URL}/api/chat", stream=False, timeout=timeout, json={
        "model": PLAN_MODEL,
        "messages": [{"role":"system","content":sys_msg},{"role":"user","content":usr_msg}]
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

def save_insight(sid, typ, content, meta=None):
    c = content.replace("'","''")
    m = json.dumps(meta or {}).replace("'","''")
    psql(f"INSERT INTO sir_insights(session_id,insight_type,content,meta) VALUES('{sid}','{typ}','{c}','{m}');")
    print(f"  [INSIGHT] {typ}: {content[:90]}")

# ── Percepcija ─────────────────────────────────────────────────────────────────
def perceive(sid):
    print("\n── PERCEPCIJA ─────────────────────────────────────────────────────")
    # Sortirano DESC po avg_delta_u_zoni — na vrhu je jezik koji NAJVIŠE profitira
    stats = psql(f"""
        SELECT lang,
               COUNT(*) as n_ukupno,
               COUNT(*) FILTER (WHERE (final_similarity - delta_similarity)
                                BETWEEN {BASELINE_LOW} AND {BASELINE_HIGH}) as u_zoni,
               ROUND(AVG(delta_similarity) FILTER (
                   WHERE (final_similarity - delta_similarity) BETWEEN {BASELINE_LOW} AND {BASELINE_HIGH}
               )::numeric, 4) as avg_delta_u_zoni,
               ROUND(AVG(delta_similarity)::numeric, 4) as avg_delta_ukupno,
               COUNT(*) FILTER (WHERE success=true) as uspjesnih
        FROM sir_trajectories
        GROUP BY lang
        ORDER BY avg_delta_u_zoni DESC NULLS LAST;
    """)
    print("Statistika (sortirano: najviši avg_delta_u_zoni = top prioritet):")
    print(stats)
    return stats

# ── Planiranje v3 ──────────────────────────────────────────────────────────────
def plan(stats, sid):
    print("\n── PLANIRANJE v3 ──────────────────────────────────────────────────")

    system = (
        "You are an orchestrator for a Self-Refine translation system. "
        "Choose the language to prioritize based on this rule:\n\n"
        "RULE: The language with the HIGHEST POSITIVE avg_delta_u_zoni gets top priority. "
        "avg_delta_u_zoni measures how much Self-Refine actually improves translations "
        "in the optimal baseline range. Positive = Self-Refine works there. "
        "Negative or null = Self-Refine doesn't help.\n\n"
        "The table is already sorted: first row = best candidate for priority.\n\n"
        "Respond with ONLY this JSON (no other text):\n"
        '{"priority_lang":"srp","reason":"one sentence","n_samples":2,"strategy":"focus_weak"}\n\n'
        "strategy options: focus_weak | balanced | test_strong\n"
        "n_samples: integer 1-3 only."
    )
    user = (
        f"Statistics (sorted by avg_delta_u_zoni DESC — first row = top candidate):\n"
        f"Columns: lang | n_ukupno | u_zoni | avg_delta_u_zoni | avg_delta_ukupno | uspjesnih\n"
        f"{stats}\n"
        "Which language should be prioritized?"
    )

    t = timer()
    raw = ""
    try:
        raw = llm_chat(system, user)
        print(f"  LLM ({elapsed(t)}): {raw}")
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        d = json.loads(m.group()) if m else {}

        lang = d.get("priority_lang","srp")
        if lang not in ("srp","hrv","bos"): lang = "srp"
        d["priority_lang"] = lang
        d["n_samples"] = max(1, min(int(d.get("n_samples",2)), MAX_N))
        d["strategy"] = str(d.get("strategy","balanced")).split("|")[0].strip()

    except Exception as e:
        print(f"  [WARN] fallback srp ({e})")
        d = {"priority_lang":"srp","reason":f"fallback:{e}","n_samples":2,"strategy":"balanced"}

    print(f"  → {d['priority_lang']} | {d['strategy']} | n={d['n_samples']} | {d.get('reason','')}")
    save_insight(sid, "planning_v3", json.dumps(d), {"raw": raw[:200]})
    return d

# ── Uzorkovanje ────────────────────────────────────────────────────────────────
def smart_sample(lang, n, sid):
    print(f"\n── UZORKOVANJE ({lang.upper()}, n={n}) ───────────────────────────────")
    raw = psql(f"""
        SELECT local_text, eng_text,
               array_length(string_to_array(eng_text,' '),1) as nw
        FROM sentence_pairs_v2
        WHERE lang='{lang}'
          AND array_length(string_to_array(eng_text,' '),1) BETWEEN {WORD_MIN} AND {WORD_MAX}
          AND id NOT IN (
              SELECT sp.id FROM sentence_pairs_v2 sp
              JOIN sir_trajectories st ON st.original_text = sp.eng_text
              WHERE sp.lang='{lang}')
        ORDER BY RANDOM() LIMIT {n};
    """)
    samples = []
    for line in raw.split('\n'):
        line = line.strip()
        if line.count('|') >= 2:
            p = [x.strip() for x in line.split('|')]
            if p[0] and p[1]:
                samples.append((p[0], p[1]))
                print(f"  [{p[2] if len(p)>2 else '?'} rij.] {p[1][:65]}")
    print(f"  → {len(samples)} uzoraka")
    return samples

# ── Self-Refine ────────────────────────────────────────────────────────────────
def run_refine(samples, lang, sid):
    sys.path.insert(0, '/home/balsam/sir/faza1')
    import self_refine_v3 as sr
    sr.PROCEDURE_ID = PROCEDURE_ID
    results = []
    for local, eng in samples:
        print(f"\n  → SR: {eng[:55]}...")
        results.append(sr.self_refine(eng, local, lang, "eng2local", sid))
    return results

# ── Opservacija ────────────────────────────────────────────────────────────────
def observe(results, decision, sid):
    print("\n── OPSERVACIJA ────────────────────────────────────────────────────")
    if not results: return
    n = len(results)
    ok = sum(1 for r in results if r.get("success"))
    lok = sum(1 for r in results if r.get("lang_ok"))
    avg_d = sum(r.get("delta",0) for r in results)/n
    avg_f = sum(r.get("final",0) for r in results)/n
    print(f"  {ok}/{n} uspješnih | lang_ok {lok}/{n} | avg_delta {avg_d:+.4f} | avg_final {avg_f:.4f}")

    summary = {"n":n,"ok":ok,"lok":lok,"avg_delta":round(avg_d,4),
               "avg_final":round(avg_f,4),"lang":decision["priority_lang"],
               "strategy":decision["strategy"]}

    per = [{"d":round(r.get("delta",0),3),"ok":r.get("success"),"lok":r.get("lang_ok")} for r in results]
    sys_r = ("You are a reflection agent. In 1-2 sentences, state the most specific "
             "pattern you observe in this Self-Refine session. Focus on what the data shows. English.")
    usr_r = f"Plan:{json.dumps(decision)}\nSummary:{json.dumps(summary)}\nPer-result:{json.dumps(per)}"
    try:
        insight = llm_chat(sys_r, usr_r, timeout=60)
        print(f"\n  [REFLEKSIJA] {insight}")
        save_insight(sid, "reflection_v3", insight, summary)
    except Exception as e:
        save_insight(sid, "reflection_v3", f"error:{e}", summary)

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = timer()
    sid = f"agent3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"  SIR Perceptivni Agent v3 | {sid}")
    print(f"  Procedure: {PROCEDURE_ID} | Max n/lang: {MAX_N}")
    print(f"  Zona: baseline {BASELINE_LOW}–{BASELINE_HIGH}, dužina {WORD_MIN}–{WORD_MAX}")
    print(f"{'='*60}")

    stats  = perceive(sid)
    dec    = plan(stats, sid)
    pri    = dec["priority_lang"]
    n      = dec["n_samples"]

    all_samples = {pri: smart_sample(pri, n, sid)}
    for lang in ["srp","hrv","bos"]:
        if lang != pri:
            all_samples[lang] = smart_sample(lang, 1, sid)

    all_results = []
    for lang, samp in all_samples.items():
        if samp:
            all_results.extend(run_refine(samp, lang, sid))

    observe(all_results, dec, sid)

    ok  = sum(1 for r in all_results if r.get("success"))
    avg = sum(r.get("delta",0) for r in all_results)/len(all_results) if all_results else 0
    print(f"\n{'='*60}")
    print(f"  Sesija: {sid} | Trajektorija: {len(all_results)}")
    print(f"  Uspješnih: {ok}/{len(all_results)} | Avg delta: {avg:+.4f}")
    print(f"  Ukupno: {elapsed(t0)}")
    print(f"{'='*60}")
