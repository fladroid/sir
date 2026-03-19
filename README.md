# SIR — Self-Improvement & Research

Istraživački projekat | Flavio & Claude | Mart 2026

## O projektu

SIR istražuje mogućnost emergencije kroz kombinaciju četiri elementa:
- **Self-Improvement** — iterativno poboljšanje modela tokom rada
- **Agentic Research** — autonomno istraživanje i integracija novih znanja
- **Skill Persistence** — pamćenje procedura u eksternoj memoriji (PostgreSQL)
- **Autonomous Execution** — rad bez čovekove potvrde za svaki korak

Centralna hipoteza: kombinacijom ova četiri elementa na Balsam projektu
(llama3.2:3b, južnoslavenski jezici) moguća je emergence — inter-jezička
hibridna strategija prevođenja koja nije bila eksplicitno programirana.

## Struktura

```
sir/
├── faza1/
│   └── self_refine.py     # Self-Refine petlja za prevod sa Skill Persistence
├── sql/
│   └── schema.sql         # PostgreSQL šema za SIR tabele
└── docs/                  # Rezervisano za buduće faze
```

## Infrastruktura

- Server: balsam.dynu.net (Ubuntu 24.04, 16GB RAM)
- Modeli: balsam:latest (llama3.2:3b, fine-tuned na srp/hrv/bos)
- Baza: PostgreSQL 17 (Docker)
- Embeddings: llama3.2:3b (dim 3072)

## Faza 1 — Self-Refine petlja

`faza1/self_refine.py` implementira osnovnu Self-Refine petlju:
1. Prevod teksta koristeći `balsam:latest`
2. Evaluacija cosine similarity prema referentnom prevodu
3. Self-critique — model identifikuje slabosti
4. Refinement — poboljšani prevod
5. Upisivanje trajektorije u `sir_trajectories`

Sve trajektorije i metapodaci se čuvaju u PostgreSQL bazi kao
eksternalizovana proceduralna memorija (Skill Persistence).

## Veza sa Balsam projektom

SIR direktno informiše Balsam Phase 2. Više na:
https://github.com/fladroid/balsam
