#!/usr/bin/env python3
"""
SIR Projekat — Faza 2
Self-Refine petlja za prevod sa Skill Persistence
Balsam server | Flavio & Claude | Mart 2026
"""

import json
import subprocess
import requests
import os
from datetime import datetime

# ── Konfiguracija ──────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api"
TRANSLATE_MODEL = "balsam:latest"
CRITIQUE_MODEL  = "balsam:latest"
EMBED_MODEL     = "llama3.2:3b"
DB_USER         = "pgu"
DB_HOST         = "127.0.0.1"
DB_NAME         = "balsam"
DB_PASS         = "Pgu1234.1234"
PROCEDURE_ID    = 1
MAX_ITERATIONS  = 3
CONV_THRESHOLD  = 0.02

# ── DB helper ──────────────────────────────────────────────────────────────────
def psql(query):
    cmd = ["docker", "exec", "pgdb",
           "psql", "-h", DB_HOST, "-U", DB_USER, "-d", DB_NAME, "-t", "-c", query]
    env = {**os.environ, "PGPASSWORD": DB_PASS}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.stdout.strip()

# ── Ollama helpers ─────────────────────────────────────────────────────────────
def translate(text, model=TRANSLATE_MODEL, critique=""):
    prompt = "Translate the following text accurately and naturally."
    if critique:
        prompt += f"\n\nPrevious attempt had these issues: {critique}\nPlease improve."
    prompt += f"\n\nText: {text}\n\nTranslation:"
    resp = requests.post(f"{OLLAMA_URL}/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.3}
    }, timeout=180)
    return resp.json().get("response", "").strip()

def get_embedding(text, model=EMBED_MODEL):
    resp = requests.post(f"{OLLAMA_URL}/embeddings", json={
        "model": model, "prompt": text
    }, timeout=30)
    return resp.json().get("embedding", [])

def cosine_similarity(v1, v2):
    if not v1 or not v2:
        return 0.0
    dot  = sum(a*b for a,b in zip(v1,v2))
    n1   = sum(a*a for a in v1)**0.5
    n2   = sum(b*b for b in v2)**0.5
    return dot/(n1*n2) if n1 and n2 else 0.0

def self_critique(original, translation, reference, similarity):
    prompt = f"""Evaluate this translation briefly.
Original: {original}
Translation: {translation}
Reference: {reference}
Score: {similarity:.3f}/1.0

List 1-2 specific issues and suggest improvements. Be concise."""
    resp = requests.post(f"{OLLAMA_URL}/generate", json={
        "model": CRITIQUE_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.2}
    }, timeout=180)
    return resp.json().get("response", "").strip()

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
    print(f"\n{'─'*60}")
    print(f"  {lang} | {direction}")
    print(f"  Orig: {original[:70]}")
    print(f"  Ref:  {reference[:70]}")

    ref_emb    = get_embedding(reference)
    steps_log  = []
    critique   = ""
    prev_sim   = 0.0

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n  [Iter {i}]")
        translation = translate(original, critique=critique)
        print(f"  → {translation[:80]}")

        t_emb = get_embedding(translation)
        sim   = cosine_similarity(t_emb, ref_emb)
        delta = sim - prev_sim
        print(f"  Similarity: {sim:.4f}  Δ{delta:+.4f}")

        step = {"iteration": i, "translation": translation,
                "similarity": round(sim,6), "delta": round(delta,6)}

        if i < MAX_ITERATIONS:
            if abs(delta) < CONV_THRESHOLD and i > 1:
                print(f"  Konvergencija — zaustavljam.")
                steps_log.append(step)
                break
            critique = self_critique(original, translation, reference, sim)
            print(f"  Kritika: {critique[:100]}")
            step["critique"] = critique

        steps_log.append(step)
        prev_sim = sim

    final_sim    = steps_log[-1]["similarity"]
    baseline_sim = steps_log[0]["similarity"]
    success      = final_sim > baseline_sim

    traj_id = save_trajectory(lang, direction, original, len(steps_log),
                               final_sim, baseline_sim, steps_log, success)
    print(f"\n  Baseline {baseline_sim:.4f} → Final {final_sim:.4f}  ({final_sim-baseline_sim:+.4f})")
    print(f"  Trajektorija ID: {traj_id}")

    return {"traj_id": traj_id, "lang": lang, "direction": direction,
            "iterations": len(steps_log), "baseline": baseline_sim,
            "final": final_sim, "delta": final_sim - baseline_sim,
            "success": success}

# ── Uzorci iz baze ─────────────────────────────────────────────────────────────
def get_samples(lang, n=2):
    rows = psql(f"""SELECT local_text, eng_text FROM sentence_pairs_v2
                    WHERE lang='{lang}' ORDER BY RANDOM() LIMIT {n};""")
    samples = []
    for line in rows.split('\n'):
        line = line.strip()
        if '|' in line:
            p = [x.strip() for x in line.split('|')]
            if len(p) >= 2 and p[0] and p[1]:
                samples.append((p[0], p[1]))
    return samples

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    session_id = f"sir_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nSIR Self-Refine | Sesija: {session_id}")
    print(f"Model: {TRANSLATE_MODEL} | Kritika: {CRITIQUE_MODEL} | Emb: {EMBED_MODEL}")

    results = []

    for lang in ["srp", "hrv", "bos"]:
        samples = get_samples(lang, n=2)
        if not samples:
            print(f"  Nema uzoraka za {lang}")
            continue
        for local_text, eng_text in samples:
            r = self_refine(eng_text, local_text, lang, "eng2local", session_id)
            results.append(r)

    # Metapodaci po jeziku
    for lang in ["srp", "hrv", "bos"]:
        lr = [r for r in results if r["lang"] == lang]
        if lr:
            avg_delta = sum(r["delta"] for r in lr) / len(lr)
            avg_final = sum(r["final"] for r in lr) / len(lr)
            save_metadata(session_id, lang, "avg_delta_similarity", avg_delta,
                         {"n": len(lr), "avg_final": avg_final})

    # Izveštaj
    print(f"\n{'='*60}")
    print(f"  SESIJA: {session_id}")
    print(f"  Trajektorija: {len(results)}")
    ok = sum(1 for r in results if r["success"])
    print(f"  Uspešnih: {ok}/{len(results)}")
    if results:
        avg = sum(r["delta"] for r in results)/len(results)
        print(f"  Prosečno Δsimilarity: {avg:+.4f}")
    print(f"{'='*60}")
