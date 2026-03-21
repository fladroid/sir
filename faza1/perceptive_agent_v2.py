#!/usr/bin/env python3
"""
SIR Projekat — Prsten 1: Perceptivni Agent v2
Ispravke:
  - Planning system prompt: jasna logika prioritizacije (više u optimalnoj zoni = prioritet)
  - n_samples cappovan na MAX_SAMPLES_PER_LANG
  - strategy parsiran ispravno (prvi token ako pipe-separated)
  - Uzorkovanje: eksplicitni baseline filter kroz quick embedding provjeru NIJE moguć
    bez embeddings za sve parove — fallback: random iz dužinske zone, što je dovoljno
Flavio & Claude | Mart 2026
"""

import json
import subprocess
import requests
import re
import time
import sys
from datetime import datetime

OLLAMA_URL         = "http://localhost:11434"
PLAN_MODEL         = "balsam:latest"
DB_USER            = "pgu"
DB_HOST            = "127.0.0.1"
DB_NAME            = "balsam"
PROCEDURE_ID       = 4
BASELINE_LOW       = 0.30
BASELINE_HIGH      = 0.70
WORD_MIN           = 5
WORD_MAX           = 20
MAX_SAMPLES_PER_LANG = 3   # tvrdi cap — agent ne može tražiti više od ovoga

def timer(): return time.time()
def elapsed(s): return f"{time.time()-s:.1f}s"

def psql(query):
    cmd = ["docker","exec","pgdb","psql","-h",DB_HOST,"-U",DB_USER,"-d",DB_NAME,"-t","-c",query]
    return subprocess.run(cmd, capture_output=True, text=True).stdout

def llm_chat(system_msg, user_msg, timeout=120):
    payload = {
        "model": PLAN_MODEL,
        "messages": [
            {"role":"system","content": system_msg},
            {"role":"user","content": user_msg}
        ],
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

def save_insight(session_id, insight_type, content, meta=None):
    content_esc = content.replace("'","''")
    meta_json = json.dumps(meta or {}).replace("'","''")
    psql(f"""INSERT INTO sir_insights (session_id, insight_type, content, meta)
             VALUES ('{session_id}','{insight_type}','{content_esc}','{meta_json}');""")
    print(f"  [INSIGHT] {insight_type}: {content[:80]}")

# ── Percepcija ─────────────────────────────────────────────────────────────────
def perceive(session_id):
    print("\n── PERCEPCIJA ─────────────────────────────────────────────────────")

    stats = psql("""
        SELECT lang,
               COUNT(*) as n_ukupno,
               ROUND(AVG(final_similarity - delta_similarity)::numeric,4) as avg_baseline,
               ROUND(AVG(delta_similarity)::numeric,4) as avg_delta,
               COUNT(*) FILTER (WHERE success=true) as uspjesnih,
               COUNT(*) FILTER (WHERE (final_similarity - delta_similarity)
                                BETWEEN 0.30 AND 0.70) as u_optimalnoj_zoni,
               ROUND(AVG(delta_similarity) FILTER (
                   WHERE (final_similarity - delta_similarity) BETWEEN 0.30 AND 0.70
               )::numeric,4) as avg_delta_u_zoni
        FROM sir_trajectories
        GROUP BY lang ORDER BY lang;
    """)
    print(stats)
    return stats

# ── Planiranje v2 — ispravljena logika ─────────────────────────────────────────
def plan(stats_text, session_id):
    print("\n── PLANIRANJE v2 ──────────────────────────────────────────────────")

    system = (
        "You are an orchestrator for a Self-Refine translation improvement system. "
        "Your job: choose which language to prioritize for the next session.\n\n"
        "PRIORITY RULE: Choose the language with the MOST trajectories in the optimal "
        "baseline zone (0.30-0.70) AND positive avg_delta_u_zoni. "
        "If two languages are tied, pick the one with higher avg_delta overall.\n\n"
        "STRATEGY RULE:\n"
        "  focus_weak   = language has avg_delta < 0 (needs work)\n"
        "  balanced     = language has avg_delta ~0 (stable, explore)\n"
        "  test_strong  = language doing well, test with harder sentences\n\n"
        "Respond with ONLY valid JSON, no other text:\n"
        '{"priority_lang": "srp", "reason": "one sentence max", '
        '"n_samples": 2, "strategy": "focus_weak"}\n\n'
        "n_samples must be an integer between 1 and 3."
    )

    user = (
        f"Statistics (columns: lang, n_ukupno, avg_baseline, avg_delta, "
        f"uspjesnih, u_optimalnoj_zoni, avg_delta_u_zoni):\n{stats_text}\n\n"
        "Which language should be prioritized and why?"
    )

    t = timer()
    try:
        raw = llm_chat(system, user, timeout=90)
        print(f"  Raw odgovor ({elapsed(t)}): {raw}")

        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            d = json.loads(match.group())
        else:
            raise ValueError("No JSON found")

        # Sanitizacija
        lang = d.get("priority_lang","srp")
        if lang not in ("srp","hrv","bos"):
            lang = "srp"
        d["priority_lang"] = lang

        # Cap n_samples
        n = int(d.get("n_samples", 2))
        d["n_samples"] = max(1, min(n, MAX_SAMPLES_PER_LANG))

        # Normalizacija strategy — uzmi prvi token ako pipe-separated
        strategy = d.get("strategy","balanced")
        d["strategy"] = strategy.split("|")[0].strip()

    except Exception as e:
        print(f"  [WARN] Planning error ({e}), fallback na srp/balanced")
        d = {"priority_lang":"srp","reason":f"fallback: {e}",
             "n_samples":2,"strategy":"balanced"}

    print(f"  → Prioritet: {d['priority_lang']} | Strategija: {d['strategy']} | "
          f"N: {d['n_samples']}")
    print(f"  → Razlog: {d.get('reason','')}")
    save_insight(session_id, "planning_decision_v2", json.dumps(d),
                 {"raw_response": raw[:200] if 'raw' in dir() else ""})
    return d

# ── Uzorkovanje ────────────────────────────────────────────────────────────────
def smart_sample(lang, n, session_id):
    print(f"\n── UZORKOVANJE ({lang.upper()}, n={n}) ───────────────────────────────")

    raw = psql(f"""
        SELECT local_text, eng_text,
               array_length(string_to_array(eng_text,' '),1) as n_words
        FROM sentence_pairs_v2
        WHERE lang='{lang}'
          AND array_length(string_to_array(eng_text,' '),1) BETWEEN {WORD_MIN} AND {WORD_MAX}
          AND id NOT IN (
              SELECT sp.id FROM sentence_pairs_v2 sp
              JOIN sir_trajectories st ON st.original_text = sp.eng_text
              WHERE sp.lang='{lang}'
          )
        ORDER BY RANDOM()
        LIMIT {n};
    """)

    samples = []
    for line in raw.split('\n'):
        line = line.strip()
        if '|' in line:
            parts = [x.strip() for x in line.split('|')]
            if len(parts) >= 2 and parts[0] and parts[1]:
                samples.append((parts[0], parts[1]))
                nw = parts[2] if len(parts)>2 else "?"
                print(f"  [{nw} rij.] {parts[1][:65]}")

    print(f"  → {len(samples)} uzoraka")
    return samples

# ── Self-Refine (uvoz iz v3) ───────────────────────────────────────────────────
def run_refine(samples, lang, session_id):
    sys.path.insert(0, '/home/balsam/sir/faza1')
    import self_refine_v3 as sr
    sr.PROCEDURE_ID = PROCEDURE_ID
    results = []
    for local_text, eng_text in samples:
        print(f"\n  → Self-Refine: {eng_text[:55]}...")
        r = sr.self_refine(eng_text, local_text, lang, "eng2local", session_id)
        results.append(r)
    return results

# ── Opservacija / Refleksija ───────────────────────────────────────────────────
def observe(results, decision, session_id):
    print("\n── OPSERVACIJA ────────────────────────────────────────────────────")
    if not results:
        print("  Nema rezultata.")
        return

    n = len(results)
    ok = sum(1 for r in results if r.get("success"))
    lok = sum(1 for r in results if r.get("lang_ok"))
    avg_delta = sum(r.get("delta",0) for r in results) / n
    avg_final = sum(r.get("final",0) for r in results) / n

    print(f"  Uspješnih: {ok}/{n} | Lang OK: {lok}/{n} | "
          f"Avg delta: {avg_delta:+.4f} | Avg final: {avg_final:.4f}")

    summary = {"n":n,"ok":ok,"lok":lok,"avg_delta":avg_delta,
               "avg_final":avg_final,"lang":decision.get("priority_lang"),
               "strategy":decision.get("strategy")}

    system = (
        "You are a reflection agent. In 1-2 sentences, state the most concrete "
        "pattern observed in this translation Self-Refine session. Be specific. English only."
    )
    user = (
        f"Plan: {json.dumps(decision)}\n"
        f"Summary: {json.dumps(summary)}\n"
        f"Per-result: {json.dumps([{'d':round(r.get('delta',0),3),'ok':r.get('success'),'lang_ok':r.get('lang_ok')} for r in results])}"
    )

    try:
        insight = llm_chat(system, user, timeout=60)
        print(f"\n  [REFLEKSIJA] {insight}")
        save_insight(session_id, "session_reflection", insight, summary)
    except Exception as e:
        print(f"  [WARN] Refleksija error: {e}")
        save_insight(session_id, "session_reflection", f"error: {e}", summary)

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    t0 = timer()
    session_id = f"agent2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"  SIR Perceptivni Agent v2 | {session_id}")
    print(f"  Max uzoraka po jeziku: {MAX_SAMPLES_PER_LANG} | Procedure: {PROCEDURE_ID}")
    print(f"  Zona: baseline {BASELINE_LOW}–{BASELINE_HIGH}, dužina {WORD_MIN}–{WORD_MAX}")
    print(f"{'='*60}")

    # 1. Percepcija
    stats = perceive(session_id)

    # 2. Planiranje
    decision = plan(stats, session_id)
    priority = decision["priority_lang"]
    n        = decision["n_samples"]

    # 3. Uzorkovanje — prioritetni jezik dobija n, ostala po 1
    all_samples = {}
    all_samples[priority] = smart_sample(priority, n, session_id)
    for lang in ["srp","hrv","bos"]:
        if lang != priority:
            all_samples[lang] = smart_sample(lang, 1, session_id)

    # 4. Self-Refine
    all_results = []
    for lang, samples in all_samples.items():
        if samples:
            res = run_refine(samples, lang, session_id)
            all_results.extend(res)

    # 5. Opservacija
    observe(all_results, decision, session_id)

    # Summary
    ok = sum(1 for r in all_results if r.get("success"))
    avg = sum(r.get("delta",0) for r in all_results)/len(all_results) if all_results else 0
    print(f"\n{'='*60}")
    print(f"  Sesija: {session_id}")
    print(f"  Trajektorija: {len(all_results)} | Uspješnih: {ok}/{len(all_results)}")
    print(f"  Avg delta: {avg:+.4f} | Ukupno: {elapsed(t0)}")
    print(f"{'='*60}")
