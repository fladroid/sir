#!/usr/bin/env python3
"""
SIR Projekat — Prsten 1: Perceptivni Agent v4
Ispravke vs v3:
  - JSON parse: agresivniji regex + retry bez newlinea
  - Refleksija: isti fix za JSON parse
  - Percepcija tablica kompaktna (bez razmaka) da model ne dodaje tekst
  - Procedure ID: 5 (isti kao v3, isti agentic flow)
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
        "stream": False,
        "messages": [{"role":"system","content":sys_msg},{"role":"user","content":usr_msg}]
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

def parse_json_robust(text):
    """Pokušava parsirati JSON iz LLM odgovora — više strategija."""
    # 1. direktno
    try: return json.loads(text)
    except: pass
    # 2. regex — uzmi samo prvu {} grupu
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    # 3. ukloni newline unutar stringa i pokušaj ponovo
    collapsed = ' '.join(text.split())
    m2 = re.search(r'\{[^{}]*\}', collapsed)
    if m2:
        try: return json.loads(m2.group())
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
        GROUP BY lang
        ORDER BY delta_zoni DESC NULLS LAST;
    """)
    print("lang | n | u_zoni | delta_zoni | delta_ukupno | ok")
    print(stats)
    return stats

# ── Planiranje v4 ──────────────────────────────────────────────────────────────
def plan(stats, sid):
    print("\n── PLANIRANJE v4 ──────────────────────────────────────────────────")

    system = (
        "You are an orchestrator for a translation Self-Refine system.\n"
        "RULE: prioritize the language with the HIGHEST POSITIVE delta_zoni.\n"
        "The table is sorted: first row = top candidate.\n"
        "Respond with ONLY one line of JSON, no newlines inside:\n"
        '{"priority_lang":"srp","reason":"short","n_samples":2,"strategy":"focus_weak"}\n'
        "n_samples: 1-3. strategy: focus_weak|balanced|test_strong."
    )
    user = (
        f"Stats (lang|n|u_zoni|delta_zoni|delta_ukupno|ok):\n{stats.strip()}\n"
        "Choose priority language. One JSON line only."
    )

    t = timer()
    raw = ""
    d = None
    try:
        raw = llm_chat(system, user)
        print(f"  LLM ({elapsed(t)}): {raw[:120]}")
        d = parse_json_robust(raw)
        if d is None: raise ValueError("JSON parse failed")

        lang = d.get("priority_lang","srp")
        if lang not in ("srp","hrv","bos"): lang = "srp"
        d["priority_lang"] = lang
        d["n_samples"] = max(1, min(int(d.get("n_samples",2)), MAX_N))
        d["strategy"] = str(d.get("strategy","balanced")).split("|")[0].strip()

    except Exception as e:
        print(f"  [WARN] fallback srp ({e})")
        d = {"priority_lang":"srp","reason":f"fallback:{e}","n_samples":2,"strategy":"balanced"}

    print(f"  → {d['priority_lang']} | {d['strategy']} | n={d['n_samples']} | {d.get('reason','')}")
    save_insight(sid, "planning_v4", json.dumps(d), {"raw": raw[:200]})
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
        print(f"\n  → SR: {eng[:60]}...")
        results.append(sr.self_refine(eng, local, lang, "eng2local", sid))
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

    summary = {"n":n,"ok":ok,"lok":lok,
               "avg_delta":round(avg_d,4),"avg_final":round(avg_f,4),
               "lang":decision["priority_lang"],"strategy":decision["strategy"]}
    per = [{"d":round(r.get("delta",0),3),"ok":r.get("success")} for r in results]

    sys_r = (
        "You are a reflection agent. State in 1-2 sentences the most specific "
        "pattern in this Self-Refine session. English only. "
        "Do NOT use JSON. Plain text response only."
    )
    usr_r = f"Plan:{json.dumps(decision)}\nSummary:{json.dumps(summary)}\nPer-result:{json.dumps(per)}"
    try:
        insight = llm_chat(sys_r, usr_r, timeout=60)
        print(f"\n  [REFLEKSIJA] {insight}")
        save_insight(sid, "reflection_v4", insight, summary)
    except Exception as e:
        save_insight(sid, "reflection_v4", f"error:{e}", summary)

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0  = timer()
    sid = f"agent4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"  SIR Perceptivni Agent v4 | {sid}")
    print(f"  Procedure: {PROCEDURE_ID} | Max n/lang: {MAX_N}")
    print(f"  Zona: baseline {BASELINE_LOW}–{BASELINE_HIGH} | dužina {WORD_MIN}–{WORD_MAX}")
    print(f"{'='*60}")

    stats = perceive(sid)
    dec   = plan(stats, sid)
    pri   = dec["priority_lang"]
    n     = dec["n_samples"]

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
    print(f"  {sid} | {len(all_results)} trajektorija | {ok}/{len(all_results)} OK")
    print(f"  Avg delta: {avg:+.4f} | Ukupno: {elapsed(t0)}")
    print(f"{'='*60}")
