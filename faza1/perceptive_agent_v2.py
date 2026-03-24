#!/usr/bin/env python3
"""
SIR Projekat — Prsten 1: Perceptivni Agent v2
Popravke: cappovan n_samples, ispravan planning prompt, baseline prefiltrar.
Flavio & Claude | Mart 2026
"""

import json
import subprocess
import requests
import time
import sys
import re
from datetime import datetime

OLLAMA_URL    = "http://localhost:11434"
PLAN_MODEL    = "balsam:latest"
DB_USER       = "pgu"
DB_HOST       = "127.0.0.1"
DB_NAME       = "balsam"
PROCEDURE_ID  = 5

BASELINE_LOW  = 0.30
BASELINE_HIGH = 0.70
WORD_MIN      = 5
WORD_MAX      = 20
N_MAX         = 4          # tvrdi cap — agent ne može tražiti više

def timer(): return time.time()
def elapsed(s): return f"{time.time()-s:.1f}s"

def psql(query):
    cmd = ["docker","exec","pgdb","psql","-h",DB_HOST,"-U",DB_USER,"-d",DB_NAME,"-t","-c",query]
    return subprocess.run(cmd, capture_output=True, text=True).stdout

def llm_chat(system_msg, user_msg, timeout=90):
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
    print(f"  [INSIGHT] {insight_type}: {content[:90]}")

# ── PERCEPCIJA ─────────────────────────────────────────────────────────────────
def perceive(session_id):
    print("\n── PERCEPCIJA ─────────────────────────────────────────────────────")
    
    stats_raw = psql(f"""
        SELECT lang,
               COUNT(*) as n_total,
               COUNT(*) FILTER (WHERE (final_similarity-delta_similarity) 
                                BETWEEN {BASELINE_LOW} AND {BASELINE_HIGH}) as n_u_zoni,
               ROUND(AVG(delta_similarity) 
                     FILTER (WHERE (final_similarity-delta_similarity) 
                             BETWEEN {BASELINE_LOW} AND {BASELINE_HIGH})::numeric,4) as avg_delta_u_zoni,
               COUNT(*) FILTER (WHERE success=true) as uspjesnih,
               ROUND(AVG(final_similarity-delta_similarity)::numeric,4) as avg_baseline_sve
        FROM sir_trajectories
        GROUP BY lang ORDER BY avg_delta_u_zoni DESC NULLS LAST;
    """)
    
    print("Statistika po jeziku (sortirano po avg_delta u optimalnoj zoni):")
    print(stats_raw)
    return stats_raw

# ── PLANIRANJE ─────────────────────────────────────────────────────────────────
def plan(stats_raw, session_id):
    print("\n── PLANIRANJE ─────────────────────────────────────────────────────")
    
    system = (
        "You are an orchestrator for a translation self-improvement pipeline. "
        "Your job: pick the language that will benefit MOST from Self-Refine in the next session. "
        "KEY RULE: Higher avg_delta_u_zoni = Self-Refine works better there = HIGHER priority. "
        "Also: languages with more trajectories in the optimal zone have more reliable statistics. "
        "n_samples must be an integer between 2 and 4. "
        "Respond ONLY with valid JSON, no other text:\n"
        '{"priority_lang":"srp|hrv|bos","reason":"one sentence max","n_samples":3,"strategy":"focus_weak"}'
    )
    
    user = (
        f"Trajectory statistics (sorted by avg_delta in optimal zone {BASELINE_LOW}–{BASELINE_HIGH}):\n"
        f"{stats_raw}\n"
        f"Columns: lang | n_total | n_u_zoni | avg_delta_u_zoni | uspjesnih | avg_baseline_sve\n"
        f"Choose priority_lang with HIGHEST avg_delta_u_zoni and sufficient n_u_zoni.\n"
        f"n_samples must be integer 2, 3, or 4."
    )
    
    t = timer()
    try:
        response = llm_chat(system, user)
        print(f"  LLM ({elapsed(t)}): {response}")
        
        json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
        if json_match:
            decision = json.loads(json_match.group())
        else:
            raise ValueError("JSON not found in response")
        
        # Tvrdi cap na n_samples
        decision["n_samples"] = max(2, min(N_MAX, int(decision.get("n_samples", 3))))
        
        # Validacija priority_lang
        if decision.get("priority_lang") not in ["srp","hrv","bos"]:
            decision["priority_lang"] = "srp"
            
    except Exception as e:
        print(f"  [WARN] Planning error ({e}), fallback na srp")
        decision = {"priority_lang":"srp","reason":f"fallback: {e}","n_samples":3,"strategy":"focus_weak"}
    
    print(f"  Odluka: lang={decision['priority_lang']}, n={decision['n_samples']}, "
          f"strategija={decision['strategy']}")
    print(f"  Razlog: {decision['reason']}")
    save_insight(session_id, "planning_decision", json.dumps(decision))
    return decision

# ── UZORKOVANJE ────────────────────────────────────────────────────────────────
def smart_sample(lang, n, session_id):
    print(f"\n── UZORKOVANJE ({lang.upper()}, n={n}) ───────────────────────────────")
    
    # Isključuje već procesovane + filtrira po dužini
    # NOVO v2: dodaje prefiltrar da ne uzima rečenice koje su vjerojatno van zone
    # (koristimo avg_baseline iz trajektorija kao heuristiku — nema direktnog baseline u bazi)
    query = f"""
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
        LIMIT {n * 3};
    """
    # Uzimamo 3x više i biramo — heuristika za buduće proširenje
    # (kad imamo embeddings za sve, moći ćemo filtrirati po predicted baseline)
    
    raw = psql(query)
    samples = []
    for line in raw.split('\n'):
        line = line.strip()
        if '|' in line:
            parts = [x.strip() for x in line.split('|')]
            if len(parts) >= 2 and parts[0] and parts[1]:
                samples.append((parts[0], parts[1]))
    
    # Uzimamo prvih n (random je već u SQL-u)
    samples = samples[:n]
    for local, eng in samples:
        nw = len(eng.split())
        print(f"  [{nw}rij.] {eng[:65]}...")
    
    print(f"  Uzeto {len(samples)}/{n} uzoraka")
    return samples

# ── SELF-REFINE (uvoz iz v3) ───────────────────────────────────────────────────
def run_refine(samples, lang, session_id):
    sys.path.insert(0, '/home/balsam/sir/faza1')
    import importlib
    import self_refine_v3 as sr
    importlib.reload(sr)
    sr.PROCEDURE_ID = PROCEDURE_ID
    
    results = []
    for local_text, eng_text in samples:
        print(f"\n  → Self-Refine: {eng_text[:55]}...")
        r = sr.self_refine(eng_text, local_text, lang, "eng2local", session_id)
        results.append(r)
    return results

# ── OPSERVACIJA ────────────────────────────────────────────────────────────────
def observe(results, decision, session_id):
    print("\n── OPSERVACIJA ────────────────────────────────────────────────────")
    if not results:
        print("  Nema rezultata.")
        return
    
    n = len(results)
    uspjesnih = sum(1 for r in results if r.get("success"))
    lang_ok_n = sum(1 for r in results if r.get("lang_ok"))
    avg_delta = sum(r.get("delta",0) for r in results) / n
    avg_final = sum(r.get("final",0) for r in results) / n
    
    print(f"  {uspjesnih}/{n} uspješnih | lang_ok {lang_ok_n}/{n} | "
          f"avg_delta {avg_delta:+.4f} | avg_final {avg_final:.4f}")
    
    summary = {"n":n,"uspjesnih":uspjesnih,"lang_ok":lang_ok_n,
               "avg_delta":round(avg_delta,4),"avg_final":round(avg_final,4),
               "priority_lang":decision["priority_lang"]}
    
    system = (
        "You are a reflection agent for a translation self-improvement system. "
        "Write ONE concrete insight about what worked or failed. "
        "Be specific: mention baseline ranges, language, pattern. "
        "Max 2 sentences. English only."
    )
    user = (
        f"Plan: {json.dumps(decision)}\n"
        f"Results: {json.dumps(summary)}\n"
        f"Per-result: {json.dumps([{'delta':round(r.get('delta',0),4),'success':r.get('success'),'baseline':round(r.get('final',0)-r.get('delta',0),4)} for r in results])}\n"
        "What pattern do you see?"
    )
    
    try:
        insight = llm_chat(system, user, timeout=60)
        print(f"  [REFLEKSIJA] {insight}")
        save_insight(session_id, "session_reflection", insight, summary)
    except Exception as e:
        save_insight(session_id, "session_reflection", f"Reflection error: {e}", summary)

# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_total = timer()
    session_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nSIR Perceptivni Agent v2 | Sesija: {session_id}")
    print(f"Procedure ID: {PROCEDURE_ID} | Zona: baseline {BASELINE_LOW}–{BASELINE_HIGH} | "
          f"Dužina: {WORD_MIN}–{WORD_MAX} rij. | Cap n_samples: {N_MAX}")
    
    # 1. Percepcija
    stats_raw = perceive(session_id)
    
    # 2. Planiranje
    decision = plan(stats_raw, session_id)
    priority = decision["priority_lang"]
    n = decision["n_samples"]
    
    # 3. Uzorkovanje — prioritetni jezik dobija n, ostali po 1
    all_results = []
    for lang in ["srp","hrv","bos"]:
        n_lang = n if lang == priority else 1
        samples = smart_sample(lang, n_lang, session_id)
        if samples:
            results = run_refine(samples, lang, session_id)
            all_results.extend(results)
    
    # 4. Opservacija
    observe(all_results, decision, session_id)
    
    # Summary
    ok = sum(1 for r in all_results if r.get("success"))
    avg = sum(r.get("delta",0) for r in all_results)/len(all_results) if all_results else 0
    print(f"\n{'='*60}")
    print(f"  AGENT v2 | {session_id}")
    print(f"  Trajektorija: {len(all_results)} | Uspješnih: {ok}/{len(all_results)}")
    print(f"  Avg delta: {avg:+.4f} | Ukupno: {elapsed(t_total)}")
    print(f"{'='*60}")
