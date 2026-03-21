#!/usr/bin/env python3
"""
SIR Projekat — Prsten 1: Perceptivni Agent
Analizira bazu, uci gdje Self-Refine pomaže, bira uzorke pametno.
Flavio & Claude | Mart 2026
"""

import json
import subprocess
import requests
import time
import sys
from datetime import datetime

# ── Konfiguracija ──────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434"
PLAN_MODEL    = "balsam:latest"
DB_USER       = "pgu"
DB_HOST       = "127.0.0.1"
DB_NAME       = "balsam"
PROCEDURE_ID  = 4          # novi procedure_id za agentic runove

# Optimalna zona učena iz trajektorija
BASELINE_LOW  = 0.30
BASELINE_HIGH = 0.70
WORD_MIN      = 5
WORD_MAX      = 20
N_SAMPLES     = 3          # po jeziku, može se promijeniti argv

# ── Helpers ────────────────────────────────────────────────────────────────────
def timer():
    return time.time()

def elapsed(s):
    return f"{time.time()-s:.1f}s"

def psql(query, print_output=False):
    cmd = ["docker","exec","pgdb","psql","-h",DB_HOST,"-U",DB_USER,"-d",DB_NAME,"-t","-c",query]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if print_output:
        print(r.stdout)
    return r.stdout

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
    print(f"  [INSIGHT] {insight_type}: {content[:80]}...")

# ── Korak 1: PERCEPCIJA — čitanje stanja iz baze ──────────────────────────────
def perceive(session_id):
    print("\n── PERCEPCIJA ─────────────────────────────────────────────────────")
    
    # Statistika po jeziku iz trajektorija
    stats_raw = psql("""
        SELECT lang,
               COUNT(*) as n,
               ROUND(AVG(final_similarity - delta_similarity)::numeric,4) as avg_baseline,
               ROUND(AVG(final_similarity)::numeric,4) as avg_final,
               ROUND(AVG(delta_similarity)::numeric,4) as avg_delta,
               COUNT(*) FILTER (WHERE success=true) as uspjesnih,
               ROUND(STDDEV(delta_similarity)::numeric,4) as std_delta,
               ROUND(AVG(final_similarity - delta_similarity) 
                     FILTER (WHERE delta_similarity > 0)::numeric,4) as avg_baseline_uspjesnih
        FROM sir_trajectories
        GROUP BY lang ORDER BY lang;
    """)
    
    # Zona gdje Self-Refine stvarno pomaže
    zone_raw = psql(f"""
        SELECT lang,
               COUNT(*) as u_zoni,
               ROUND(AVG(delta_similarity)::numeric,4) as avg_delta_u_zoni
        FROM sir_trajectories
        WHERE (final_similarity - delta_similarity) BETWEEN {BASELINE_LOW} AND {BASELINE_HIGH}
        GROUP BY lang ORDER BY lang;
    """)
    
    print("Statistika trajektorija po jeziku:")
    print(stats_raw)
    print(f"Trajektorije u optimalnoj zoni ({BASELINE_LOW}–{BASELINE_HIGH}):")
    print(zone_raw)
    
    return {"stats": stats_raw, "zone": zone_raw}

# ── Korak 2: PLANIRANJE — LLM odlučuje prioritete ─────────────────────────────
def plan(perception, session_id):
    print("\n── PLANIRANJE ─────────────────────────────────────────────────────")
    
    system = (
        "You are an orchestrator for a translation self-improvement system. "
        "Analyze the provided statistics and decide which language to prioritize "
        "for the next Self-Refine session. "
        "Respond in exactly this JSON format (no other text): "
        '{"priority_lang": "srp|hrv|bos", "reason": "one sentence", '
        '"n_samples": 3, "strategy": "focus_weak|balanced|test_strong"}'
    )
    
    user = (
        f"Trajectory statistics by language:\n{perception['stats']}\n\n"
        f"Trajectories in optimal baseline zone ({BASELINE_LOW}–{BASELINE_HIGH}):\n{perception['zone']}\n\n"
        f"Optimal Self-Refine zone: baseline {BASELINE_LOW}–{BASELINE_HIGH}, "
        f"word count {WORD_MIN}–{WORD_MAX}. "
        "High baseline (>0.85) = model already good, refine doesn't help. "
        "Which language should we prioritize and why?"
    )
    
    t = timer()
    try:
        response = llm_chat(system, user)
        print(f"  Agent odgovor ({elapsed(t)}): {response}")
        
        # Parsiranje JSON odgovora
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            decision = json.loads(json_match.group())
        else:
            # Fallback ako JSON nije parsiran
            decision = {"priority_lang": "srp", "reason": "fallback — JSON parse failed", 
                        "n_samples": N_SAMPLES, "strategy": "focus_weak"}
            print(f"  [WARN] JSON parse neuspješan, fallback na srp")
    except Exception as e:
        decision = {"priority_lang": "srp", "reason": f"error: {e}",
                    "n_samples": N_SAMPLES, "strategy": "focus_weak"}
        print(f"  [WARN] LLM planning error: {e}, fallback na srp")
    
    print(f"  Odluka: prioritet={decision.get('priority_lang')}, "
          f"strategija={decision.get('strategy')}, "
          f"n={decision.get('n_samples',N_SAMPLES)}")
    print(f"  Razlog: {decision.get('reason','')}")
    
    save_insight(session_id, "planning_decision", 
                 json.dumps(decision), 
                 {"perception_summary": perception['stats'][:200]})
    
    return decision

# ── Korak 3: AKCIJA — pametno uzorkovanje ─────────────────────────────────────
def smart_sample(lang, n, session_id):
    """
    Uzima n uzoraka iz optimalne zone:
    - dužina 5-20 riječi
    - za lang gdje imamo embeddings: isključuje već procesovane
    - random unutar zone (ne sasvim random kao v3)
    """
    print(f"\n── UZORKOVANJE ({lang.upper()}, n={n}) ───────────────────────────────")
    
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
        LIMIT {n};
    """
    
    raw = psql(query)
    samples = []
    for line in raw.split('\n'):
        line = line.strip()
        if '|' in line:
            parts = [x.strip() for x in line.split('|')]
            if len(parts) >= 2 and parts[0] and parts[1]:
                local_text, eng_text = parts[0], parts[1]
                samples.append((local_text, eng_text))
                n_words = parts[2] if len(parts) > 2 else "?"
                print(f"  Uzorak ({n_words} rij.): {eng_text[:60]}...")
    
    print(f"  Uzeto {len(samples)} uzoraka iz optimalne zone")
    return samples

# ── Korak 4: UVOZ self_refine logike ──────────────────────────────────────────
def run_self_refine_for_samples(samples, lang, session_id):
    """
    Direktno uvozi i poziva self_refine iz v3 skripte.
    """
    sys.path.insert(0, '/home/balsam/sir/faza1')
    
    # Monkey-patch procedure ID u v3 modulu
    import importlib
    import self_refine_v3 as sr
    sr.PROCEDURE_ID = PROCEDURE_ID
    
    results = []
    for local_text, eng_text in samples:
        print(f"\n  Pokrećem Self-Refine za: {eng_text[:50]}...")
        r = sr.self_refine(eng_text, local_text, lang, "eng2local", session_id)
        results.append(r)
    
    return results

# ── Korak 5: OPSERVACIJA — agent gleda rezultate ──────────────────────────────
def observe(results, decision, session_id):
    print("\n── OPSERVACIJA ────────────────────────────────────────────────────")
    
    if not results:
        print("  Nema rezultata za analizu.")
        return
    
    summary = {
        "n": len(results),
        "uspjesnih": sum(1 for r in results if r.get("success")),
        "lang_ok": sum(1 for r in results if r.get("lang_ok")),
        "avg_delta": sum(r.get("delta",0) for r in results)/len(results),
        "avg_final": sum(r.get("final",0) for r in results)/len(results),
        "priority_lang": decision.get("priority_lang"),
        "strategy": decision.get("strategy")
    }
    
    print(f"  Rezultati: {summary['uspjesnih']}/{summary['n']} uspješnih")
    print(f"  Lang OK: {summary['lang_ok']}/{summary['n']}")
    print(f"  Avg delta: {summary['avg_delta']:+.4f}")
    print(f"  Avg final: {summary['avg_final']:.4f}")
    
    # LLM refleksija
    system = (
        "You are a reflection agent for a translation self-improvement system. "
        "Analyze the session results and write ONE concrete insight about what worked "
        "or didn't work. Be specific about patterns. Max 2 sentences. In English."
    )
    user = (
        f"Session plan: {json.dumps(decision)}\n"
        f"Results: {json.dumps(summary)}\n"
        f"Individual results: {json.dumps([{'delta':r.get('delta',0),'success':r.get('success'),'lang_ok':r.get('lang_ok')} for r in results])}\n"
        "What pattern do you observe?"
    )
    
    try:
        insight_text = llm_chat(system, user, timeout=60)
        print(f"\n  [REFLEKSIJA] {insight_text}")
        save_insight(session_id, "session_reflection", insight_text, summary)
    except Exception as e:
        print(f"  [WARN] Refleksija neuspješna: {e}")
        save_insight(session_id, "session_reflection", 
                    f"Reflection failed: {e}", summary)

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else N_SAMPLES
    
    t_total = timer()
    session_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nSIR Perceptivni Agent | Sesija: {session_id}")
    print(f"N uzoraka po jeziku: {n_arg} | Procedure ID: {PROCEDURE_ID}")
    print(f"Optimalna zona: baseline {BASELINE_LOW}–{BASELINE_HIGH}, "
          f"dužina {WORD_MIN}–{WORD_MAX} rij.")
    
    # 1. Percepcija
    perception = perceive(session_id)
    
    # 2. Planiranje
    decision = plan(perception, session_id)
    
    priority_lang = decision.get("priority_lang", "srp")
    n = decision.get("n_samples", n_arg)
    
    # 3. Uzorkovanje — prioritetni jezik + ostala dva (1 uzorak)
    all_samples = {}
    all_samples[priority_lang] = smart_sample(priority_lang, n, session_id)
    for lang in ["srp","hrv","bos"]:
        if lang != priority_lang:
            all_samples[lang] = smart_sample(lang, 1, session_id)
    
    # 4. Self-Refine
    all_results = []
    for lang, samples in all_samples.items():
        if samples:
            results = run_self_refine_for_samples(samples, lang, session_id)
            all_results.extend(results)
    
    # 5. Opservacija
    observe(all_results, decision, session_id)
    
    # Finalni summary
    print(f"\n{'='*60}")
    print(f"  AGENT SESIJA: {session_id}")
    print(f"  Trajektorija: {len(all_results)}")
    ok = sum(1 for r in all_results if r.get("success"))
    print(f"  Uspješnih: {ok}/{len(all_results)}")
    if all_results:
        avg = sum(r.get("delta",0) for r in all_results)/len(all_results)
        print(f"  Avg delta: {avg:+.4f}")
    print(f"  Ukupno vrijeme: {elapsed(t_total)}")
    print(f"{'='*60}")
