#!/usr/bin/env python3
"""
SIR Projekat — Prsten 3: Generativni Agent (Meta-Optimizer)
============================================================
Prsten 1 percipira i izvršava.
Prsten 2 reflektuje i sintetiše.
Prsten 3 GENERIŠE PRIJEDLOGE ZA POBOLJŠANJE samog pipeline-a.

Petlja:
  ANALYSE  → čita trajectories + insights, traži sistematske obrasce
  GENERATE → generiše kandidat-procedure (klase A i B)
  VALIDATE → pokreće mini-run (n=5) s kandidatom, mjeri avg_delta
  DECIDE   → promoviše ili odbacuje, sve bilježi u sir_candidate_procedures

Klasa A: parametarski prijedlozi (threshold, iterations, zone)   — samo JSON
Klasa B: prompt prijedlozi (system_prompt, critique instrukcija)  — novi tekst
Klasa C: arhitekturni prijedlozi (deferred)                       — samo bilježi

Flavio & Claude | Mart 2026
"""

import json, subprocess, requests, re, time, math, sys, os
from datetime import datetime

OLLAMA_URL    = "http://localhost:11434"
EMBED_MODEL   = "llama3.2:3b"
ANALYSE_MODEL = "balsam:latest"

sys.path.insert(0, '/home/balsam/fleet')
import orchestrator as fleet

DB_USER, DB_HOST, DB_NAME = "pgu", "127.0.0.1", "balsam"
DB_PASS = "Pgu1234.1234"

PARENT_PROC_ID     = 5
VALIDATION_N       = 5
BASELINE_LOW       = 0.30
BASELINE_HIGH      = 0.70
WORD_MIN, WORD_MAX = 5, 20
PROMOTE_THRESHOLD  = 0.05

def timer(): return time.time()
def elapsed(s): return f"{time.time()-s:.1f}s"

def psql(q):
    cmd = ["docker","exec","pgdb","psql","-h",DB_HOST,"-U",DB_USER,"-d",DB_NAME,"-t","-c",q]
    env = {**os.environ, "PGPASSWORD": DB_PASS}
    return subprocess.run(cmd, capture_output=True, text=True, env=env).stdout

def llm_chat(sys_msg, usr_msg, timeout=120):
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=timeout, json={
        "model": ANALYSE_MODEL, "stream": False,
        "messages": [{"role":"system","content":sys_msg},
                     {"role":"user","content":usr_msg}]
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
        m = re.search(r'\{.*\}', attempt, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return None

def banner(msg):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")

def analyse():
    banner("FAZA 1: ANALIZA")
    lang_stats = psql("""
        SELECT lang,
               COUNT(*) as n,
               ROUND(AVG(delta_similarity)::numeric,4) as avg_delta,
               ROUND(AVG(delta_similarity) FILTER (
                   WHERE (final_similarity-delta_similarity) BETWEEN 0.30 AND 0.70
               )::numeric,4) as avg_delta_zona,
               ROUND(STDDEV(delta_similarity)::numeric,4) as std_delta,
               COUNT(*) FILTER (WHERE delta_similarity < 0) as regresije,
               ROUND(AVG(iterations)::numeric,2) as avg_iter
        FROM sir_trajectories WHERE procedure_id=5
        GROUP BY lang ORDER BY avg_delta_zona ASC NULLS LAST;
    """)
    print("Statistika (procedure_id=5, sortirano po avg_delta_zona ASC):")
    print(lang_stats)

    bos_detail = psql("""
        SELECT
          ROUND(AVG(final_similarity-delta_similarity)::numeric,4) as avg_baseline,
          ROUND(AVG(delta_similarity)::numeric,4) as avg_delta,
          COUNT(*) FILTER (WHERE delta_similarity < 0) as regresije,
          COUNT(*) FILTER (WHERE delta_similarity = 0) as nula,
          COUNT(*) FILTER (WHERE delta_similarity > 0) as poboljsanja,
          ROUND(AVG(iterations)::numeric,2) as avg_iter
        FROM sir_trajectories WHERE lang='bos';
    """)
    print("\nBOS detalj (sve procedure):")
    print(bos_detail)

    insights = psql("""
        SELECT content FROM sir_insights
        WHERE insight_type LIKE 'cross_session%'
        ORDER BY created_at DESC LIMIT 3;
    """)
    print("\nPosljednji cross-session insights:")
    print(insights)

    return {"lang_stats": lang_stats.strip(),
            "bos_detail": bos_detail.strip(),
            "insights": insights.strip()}

def generate(analysis_data):
    banner("FAZA 2: GENERACIJA HIPOTEZA")

    system = """You are a Meta-Optimizer for a Self-Refine translation system (Serbian/Croatian/Bosnian).
You analyse performance statistics and generate concrete improvement hypotheses.

For each hypothesis output ONE JSON object. Generate 2-3 hypotheses.
Klasa A = parameter change (threshold, iterations, zone boundaries)
Klasa B = prompt change (different instruction text for translate or critique)

JSON format for each hypothesis:
{
  "klasa": "A",
  "lang_target": "bos",
  "hypothesis": "one sentence explaining what you think is wrong and why this fix helps",
  "proposed_config": {
    "param": "early_stopping_threshold",
    "current_value": 0.02,
    "proposed_value": 0.01,
    "rationale": "bos has lower absolute similarity values, 0.02 causes premature stop"
  }
}

For klasa B, proposed_config must include "prompt_role" (translate|critique),
"current_instruction" (summary), "proposed_instruction" (full new text).
Output: JSON array of 2-3 hypothesis objects. No preamble, no markdown."""

    # Skraćujemo insights da ne preduljimo prompt (model je mali, 3B)
    insights_trimmed = analysis_data['insights'][:400] if analysis_data['insights'] else "none"
    user = (f"LANG STATS (sorted worst first):\n{analysis_data['lang_stats'][:500]}\n\n"
            f"BOSNIAN DETAIL:\n{analysis_data['bos_detail'][:300]}\n\n"
            f"KEY INSIGHTS:\n{insights_trimmed}\n\n"
            f"Generate 2 hypotheses for worst-performing language. JSON array only.")

    t = timer()
    raw = llm_chat(system, user, timeout=180)
    print(f"LLM ({elapsed(t)}):\n{raw[:600]}")

    hypotheses = []
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"```[a-z]*\n?", "", cleaned).strip()
        parsed = json.loads(cleaned)
        hypotheses = parsed if isinstance(parsed, list) else [parsed]
    except Exception as e:
        print(f"  [WARN] JSON array parse neuspješan ({e}), pokušavam regex...")
        for m in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', raw, re.DOTALL):
            try:
                obj = json.loads(m.group())
                if "klasa" in obj and "hypothesis" in obj:
                    hypotheses.append(obj)
            except: pass

    if not hypotheses:
        print("  [WARN] Fallback hipoteze (LLM nije dao validne)")
        hypotheses = [
            {
                "klasa": "A", "lang_target": "bos",
                "hypothesis": "bos early_stopping_threshold 0.02 prevelik za visoke baseline vrijednosti (~0.88), uzrokuje preuranjeni stop.",
                "proposed_config": {"param": "early_stopping_threshold",
                                    "current_value": 0.02, "proposed_value": 0.01,
                                    "rationale": "bos avg_baseline ~0.88, abs delta inkrementi manji nego srp"}
            },
            {
                "klasa": "A", "lang_target": "bos",
                "hypothesis": "bos max_iterations=3 nedovoljno za izlaz iz lokalnog minimuma — avg_iter=2.0 znaci da staje prerano.",
                "proposed_config": {"param": "max_iterations",
                                    "current_value": 3, "proposed_value": 5,
                                    "rationale": "bos konvergira na iter 2 s nula deltom — treba vise pokusaja"}
            }
        ]

    print(f"\n  Generirano {len(hypotheses)} hipoteza:")
    for i, h in enumerate(hypotheses):
        print(f"  [{i+1}] Klasa {h.get('klasa','?')} | {h.get('lang_target','?')} | {h.get('hypothesis','')[:80]}")
    return hypotheses

def validate_hypothesis(hyp, baseline_delta, sid):
    lang = hyp.get("lang_target", "bos")
    config = hyp.get("proposed_config", {})
    klasa = hyp.get("klasa", "A")

    banner(f"FAZA 3: VALIDACIJA | Klasa {klasa} | {lang.upper()}")
    print(f"  Hipoteza: {hyp.get('hypothesis','')[:100]}")
    print(f"  Config: {json.dumps(config)[:150]}")

    raw_samples = psql(f"""
        SELECT local_text, eng_text
        FROM sentence_pairs_v2
        WHERE lang='{lang}'
          AND array_length(string_to_array(eng_text,' '),1) BETWEEN {WORD_MIN} AND {WORD_MAX}
          AND id NOT IN (
              SELECT sp.id FROM sentence_pairs_v2 sp
              JOIN sir_trajectories st ON st.original_text = sp.eng_text
              WHERE sp.lang='{lang}')
        ORDER BY RANDOM() LIMIT {VALIDATION_N * 3};
    """)

    candidates = []
    for line in raw_samples.split('\n'):
        line = line.strip()
        if '|' in line:
            p = [x.strip() for x in line.split('|')]
            if len(p) >= 2 and p[0] and p[1]:
                candidates.append((p[0], p[1]))

    samples_in_zone = []
    print(f"  Pre-scoring {len(candidates)} kandidata...")
    for local, eng in candidates:
        if len(samples_in_zone) >= VALIDATION_N: break
        try:
            sim = cosine(get_embedding(eng), get_embedding(local))
            if BASELINE_LOW <= sim <= BASELINE_HIGH:
                samples_in_zone.append((local, eng))
                print(f"    ✓ {sim:.3f} | {eng[:50]}")
        except: pass

    if len(samples_in_zone) < 2:
        print(f"  [WARN] Premalo uzoraka u zoni ({len(samples_in_zone)})")
        return None

    print(f"  Uzoraka za validaciju: {len(samples_in_zone)}")

    sys.path.insert(0, '/home/balsam/sir/faza1')
    import self_refine_v3 as sr
    import importlib
    importlib.reload(sr)

    if klasa == "A":
        param = config.get("param", "")
        val   = config.get("proposed_value")
        if param == "early_stopping_threshold" and val is not None:
            sr.CONV_THRESHOLD = float(val)
            print(f"  Override: CONV_THRESHOLD = {sr.CONV_THRESHOLD}")
        elif param == "max_iterations" and val is not None:
            sr.MAX_ITERATIONS = int(val)
            print(f"  Override: MAX_ITERATIONS = {sr.MAX_ITERATIONS}")

    if klasa == "B":
        print("  [INFO] Klasa B — bilježim prompt prijedlog, validacija s trenutnim promptom")

    sr.PROCEDURE_ID = PARENT_PROC_ID
    t_start = timer()
    results = []
    for local, eng in samples_in_zone:
        try:
            print(f"\n    SR: {eng[:60]}...")
            res = sr.self_refine(eng, local, lang, "eng2local", sid)
            results.append(res)
            print(f"    delta={res.get('delta',0):+.4f} | iter={res.get('iterations',0)} | ok={res.get('success')}")
        except Exception as e:
            print(f"    [ERR] {e}")

    if not results: return None
    avg_d = sum(r.get("delta", 0) for r in results) / len(results)
    ok    = sum(1 for r in results if r.get("success"))
    print(f"\n  Validacija završena ({elapsed(t_start)})")
    print(f"  avg_delta={avg_d:+.4f} | baseline={baseline_delta:+.4f} | poboljšanje={avg_d-baseline_delta:+.4f}")
    print(f"  OK: {ok}/{len(results)}")
    return {"validation_delta": round(avg_d,4), "validation_n": len(results),
            "baseline_delta": round(baseline_delta,4), "improvement": round(avg_d-baseline_delta,4)}

def decide_and_save(hyp, val_result, sid):
    banner("FAZA 4: ODLUKA")
    lang  = hyp.get("lang_target", "bos")
    klasa = hyp.get("klasa", "A")
    hypothesis = hyp.get("hypothesis", "")
    config = hyp.get("proposed_config", {})

    if val_result is None:
        promoted = False
        notes = "Validacija nije mogla biti izvršena — nedovoljno uzoraka u zoni"
        v_delta = v_n = b_delta = None
        print("  → NIJE PROMOVISANO (nema validacije)")
    else:
        improvement = val_result.get("improvement", 0)
        promoted = improvement >= PROMOTE_THRESHOLD
        v_delta, v_n, b_delta = val_result["validation_delta"], val_result["validation_n"], val_result["baseline_delta"]
        if promoted:
            notes = f"Promovisano: poboljšanje {improvement:+.4f} > threshold {PROMOTE_THRESHOLD}"
            print(f"  → PROMOVISANO ✓ | poboljšanje={improvement:+.4f}")
        else:
            notes = f"Odbačeno: poboljšanje {improvement:+.4f} < threshold {PROMOTE_THRESHOLD}"
            print(f"  → ODBAČENO ✗ | poboljšanje={improvement:+.4f} (threshold={PROMOTE_THRESHOLD})")

    hyp_esc    = hypothesis.replace("'","''")
    config_json = json.dumps(config).replace("'","''")
    notes_esc  = notes.replace("'","''")
    v_delta_sql = str(v_delta) if v_delta is not None else "NULL"
    v_n_sql     = str(v_n)     if v_n     is not None else "NULL"
    b_delta_sql = str(b_delta) if b_delta is not None else "NULL"

    psql(f"""
        INSERT INTO sir_candidate_procedures
            (parent_proc_id, klasa, lang_target, hypothesis,
             proposed_config, validation_delta, validation_n,
             baseline_delta, promoted, notes)
        VALUES
            ({PARENT_PROC_ID}, '{klasa}', '{lang}', '{hyp_esc}',
             '{config_json}', {v_delta_sql}, {v_n_sql},
             {b_delta_sql}, {str(promoted).upper()}, '{notes_esc}');
    """)
    print("  Zapisano u sir_candidate_procedures")

    if promoted:
        proc_name = f"gen_opt_{lang}_{klasa.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        proc_desc = (f"Auto-generisana procedura od Prstena 3. Klasa {klasa} | {lang} | "
                     f"Hipoteza: {hypothesis[:100]}. "
                     f"Validacija avg_delta={v_delta:+.4f} vs baseline={b_delta:+.4f}.")
        new_id = psql(f"""
            INSERT INTO sir_procedures(name, description)
            VALUES('{proc_name.replace("'","''")}', '{proc_desc.replace("'","''")}')
            RETURNING id;
        """).strip()
        print(f"  Nova procedura upisana (id={new_id})")
        new_id_clean = new_id.strip().split()[-1] if new_id.strip() else ''
        if new_id_clean and new_id_clean.isdigit():
            new_id = new_id_clean
            psql(f"""UPDATE sir_candidate_procedures SET promoted_proc_id={new_id.strip()}
                     WHERE lang_target='{lang}' AND klasa='{klasa}'
                       AND hypothesis='{hyp_esc}' AND promoted=TRUE
                       AND created_at > NOW() - INTERVAL '1 minute';""")
    return promoted

def print_summary():
    banner("SUMMARY — sir_candidate_procedures (zadnjih 10 min)")
    summary = psql("""
        SELECT klasa, lang_target, promoted,
               ROUND(validation_delta::numeric,4) as val_delta,
               ROUND(baseline_delta::numeric,4) as base_delta,
               ROUND((validation_delta-baseline_delta)::numeric,4) as poboljsanje,
               LEFT(hypothesis,70) as hipoteza
        FROM sir_candidate_procedures
        WHERE created_at > NOW() - INTERVAL '10 minutes'
        ORDER BY created_at;
    """)
    print(summary)

if __name__ == "__main__":
    t0  = timer()
    sid = f"gen3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n{'='*60}")
    print(f"  SIR Generativni Agent — Prsten 3 (Meta-Optimizer)")
    print(f"  Sesija: {sid}")
    print(f"  Parent procedure: {PARENT_PROC_ID} | Promote threshold: {PROMOTE_THRESHOLD}")
    print(f"  Validacija: {VALIDATION_N} uzoraka po hipotezi")
    print(f"{'='*60}")

    baseline_row = psql("""
        SELECT ROUND(AVG(delta_similarity)::numeric,4)
        FROM sir_trajectories WHERE procedure_id=5 AND lang='bos';
    """).strip()
    baseline_delta = float(baseline_row) if baseline_row.replace('.','').replace('-','').isdigit() else 0.001
    print(f"\nBaseline avg_delta (bos, proc=5): {baseline_delta:+.4f}")

    analysis   = analyse()
    hypotheses = generate(analysis)

    promoted_count = 0
    for i, hyp in enumerate(hypotheses):
        print(f"\n{'='*60}")
        print(f"  Hipoteza {i+1}/{len(hypotheses)}: Klasa {hyp.get('klasa','?')} | {hyp.get('lang_target','?')}")
        print(f"{'='*60}")
        val_result = validate_hypothesis(hyp, baseline_delta, sid)
        promoted   = decide_and_save(hyp, val_result, sid)
        if promoted: promoted_count += 1

    print_summary()
    print(f"\n{'='*60}")
    print(f"  Prsten 3 završen | {elapsed(t0)}")
    print(f"  Hipoteze: {len(hypotheses)} | Promovisano: {promoted_count}")
    print(f"  Sesija: {sid}")
    print(f"{'='*60}")
