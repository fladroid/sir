#!/usr/bin/env python3
"""
SIR Projekat — Prsten 2: Refleksivni Agent
Čita sir_insights i sir_trajectories iz više sesija, detektira
obrasce koji se ponavljaju, i upisuje strukturirane zaključke
u sir_insights (tip: cross_session_pattern).
Ovi zaključci informiraju buduće planiranje Prstena 1.
Flavio & Claude | Mart 2026
"""

import json, subprocess, requests, re, time
from datetime import datetime

OLLAMA_URL = "http://localhost:11434"
REFLECT_MODEL = "balsam:latest"
DB_USER, DB_HOST, DB_NAME = "pgu", "127.0.0.1", "balsam"

def timer(): return time.time()
def elapsed(s): return f"{time.time()-s:.1f}s"

def psql(q):
    cmd = ["docker","exec","pgdb","psql","-h",DB_HOST,"-U",DB_USER,"-d",DB_NAME,"-t","-c",q]
    return subprocess.run(cmd, capture_output=True, text=True).stdout

def llm_chat(sys_msg, usr_msg, timeout=120):
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=timeout, json={
        "model": REFLECT_MODEL, "stream": False,
        "messages": [{"role":"system","content":sys_msg},{"role":"user","content":usr_msg}]
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

def save_pattern(pattern_type, content, meta=None):
    sid = f"reflect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    c = content.replace("'","''")
    m = json.dumps(meta or {}).replace("'","''")
    psql(f"INSERT INTO sir_insights(session_id,insight_type,content,meta) "
         f"VALUES('{sid}','cross_session_{pattern_type}','{c}','{m}');")
    print(f"  [PATTERN:{pattern_type}] {content[:100]}")

# ── Analiza 1: statistika po jeziku kroz sve sesije ───────────────────────────
def analyse_lang_trends():
    print("\n── ANALIZA: Trendovi po jeziku ────────────────────────────────────")
    raw = psql("""
        SELECT lang,
               COUNT(*) as n_total,
               ROUND(AVG(final_similarity - delta_similarity)::numeric,4) as avg_baseline,
               ROUND(AVG(final_similarity)::numeric,4) as avg_final,
               ROUND(AVG(delta_similarity)::numeric,4) as avg_delta,
               ROUND(STDDEV(delta_similarity)::numeric,4) as std_delta,
               COUNT(*) FILTER (WHERE success=true) as n_ok,
               COUNT(*) FILTER (WHERE (final_similarity-delta_similarity)
                                BETWEEN 0.30 AND 0.70) as n_u_zoni,
               ROUND(AVG(delta_similarity) FILTER (
                   WHERE (final_similarity-delta_similarity) BETWEEN 0.30 AND 0.70
               )::numeric,4) as avg_delta_u_zoni
        FROM sir_trajectories
        GROUP BY lang ORDER BY avg_delta DESC;
    """)
    print(raw)
    return raw

# ── Analiza 2: trend kroz procedure (v1→v2→v3→agentic) ───────────────────────
def analyse_procedure_trend():
    print("\n── ANALIZA: Trend po proceduri (verziji skripte) ──────────────────")
    raw = psql("""
        SELECT p.name, p.id as proc_id,
               COUNT(t.id) as n,
               ROUND(AVG(t.delta_similarity)::numeric,4) as avg_delta,
               COUNT(t.id) FILTER (WHERE t.success=true) as n_ok
        FROM sir_procedures p
        JOIN sir_trajectories t ON t.procedure_id = p.id
        GROUP BY p.id, p.name
        ORDER BY p.id;
    """)
    print(raw)
    return raw

# ── Analiza 3: refleksije sesija (plain text iz sir_insights) ─────────────────
def gather_reflections():
    print("\n── ANALIZA: Sesijske refleksije ───────────────────────────────────")
    raw = psql("""
        SELECT session_id, insight_type, content
        FROM sir_insights
        WHERE insight_type LIKE '%reflection%'
          AND content NOT LIKE 'error:%'
        ORDER BY created_at;
    """)
    print(raw)
    return raw

# ── Analiza 4: best_so_far intervencije ──────────────────────────────────────
def analyse_best_so_far():
    print("\n── ANALIZA: Distribucija po iteracijama (gdje je best?) ────────────")
    raw = psql("""
        SELECT
          COUNT(*) as ukupno,
          COUNT(*) FILTER (WHERE delta_similarity > 0.1) as veliki_pomak,
          COUNT(*) FILTER (WHERE delta_similarity BETWEEN 0 AND 0.1) as mali_pomak,
          COUNT(*) FILTER (WHERE delta_similarity < 0) as regresija,
          ROUND(MAX(delta_similarity)::numeric,4) as max_delta,
          ROUND(MIN(delta_similarity)::numeric,4) as min_delta
        FROM sir_trajectories;
    """)
    print(raw)

    # Rečenice s najvećim poboljšanjem — što su imale zajedničkog?
    best = psql("""
        SELECT lang, original_text,
               ROUND((final_similarity-delta_similarity)::numeric,4) as baseline,
               ROUND(final_similarity::numeric,4) as final,
               ROUND(delta_similarity::numeric,4) as delta
        FROM sir_trajectories
        WHERE delta_similarity > 0.3
        ORDER BY delta_similarity DESC LIMIT 8;
    """)
    print("Top trajektorije (delta > 0.3):")
    print(best)
    return raw, best

# ── LLM sinteza: cross-session pattern ───────────────────────────────────────
def synthesize(lang_trends, proc_trend, reflections, dist_raw, best_raw):
    print("\n── LLM SINTEZA ─────────────────────────────────────────────────────")

    system = (
        "You are a meta-analyst for a Self-Refine translation system. "
        "You have access to statistics from multiple sessions. "
        "Your task: identify 2-3 concrete, actionable patterns that should "
        "guide future sessions. Be specific — reference actual numbers. "
        "Format: numbered list, plain English, no JSON."
    )
    user = (
        f"LANGUAGE TRENDS:\n{lang_trends}\n\n"
        f"PROCEDURE TREND (versions over time):\n{proc_trend}\n\n"
        f"SESSION REFLECTIONS:\n{reflections}\n\n"
        f"DISTRIBUTION (success/fail/regression):\n{dist_raw}\n\n"
        f"TOP IMPROVEMENTS (delta > 0.3):\n{best_raw}\n\n"
        "What are the 2-3 most important patterns across all sessions?"
    )

    t = timer()
    try:
        synthesis = llm_chat(system, user, timeout=120)
        print(f"\n  LLM ({elapsed(t)}):\n{synthesis}")
        return synthesis
    except Exception as e:
        print(f"  [WARN] Sinteza error: {e}")
        return f"error:{e}"

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = timer()
    print(f"\n{'='*60}")
    print(f"  SIR Refleksivni Agent (Prsten 2)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    lang_trends = analyse_lang_trends()
    proc_trend  = analyse_procedure_trend()
    reflections = gather_reflections()
    dist_raw, best_raw = analyse_best_so_far()

    synthesis = synthesize(lang_trends, proc_trend, reflections, dist_raw, best_raw)

    # Upiši u bazu kao cross-session pattern
    save_pattern("synthesis", synthesis, {
        "n_trajectories": psql("SELECT COUNT(*) FROM sir_trajectories;").strip(),
        "n_sessions": psql("SELECT COUNT(DISTINCT session_id) FROM sir_insights;").strip(),
        "generated_at": datetime.now().isoformat()
    })

    # Konkretni, brojčani zaključci za planiranje
    print("\n── NUMERIČKI ZAKLJUČCI ─────────────────────────────────────────────")
    conclusions = psql("""
        SELECT
            (SELECT lang FROM sir_trajectories
             WHERE (final_similarity-delta_similarity) BETWEEN 0.30 AND 0.70
             GROUP BY lang ORDER BY AVG(delta_similarity) DESC LIMIT 1)
             as best_lang_u_zoni,
            (SELECT ROUND(AVG(delta_similarity)::numeric,4) FROM sir_trajectories
             WHERE (final_similarity-delta_similarity) BETWEEN 0.30 AND 0.70
               AND delta_similarity > 0) as avg_delta_pozitivnih_u_zoni,
            (SELECT COUNT(*) FROM sir_trajectories WHERE delta_similarity > 0.3) as veliki_uspjesi,
            (SELECT COUNT(*) FROM sir_trajectories) as ukupno;
    """)
    print(conclusions)

    print(f"\n{'='*60}")
    print(f"  Refleksija završena | {elapsed(t0)}")
    print(f"  Upisan cross_session_synthesis u sir_insights")
    print(f"{'='*60}")
