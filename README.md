# Analyseur Intelligent de CV (Streamlit)

Application web Streamlit qui analyse un CV (PDF) par rapport à une offre d’emploi : score ATS (similarité sémantique), rapport IA détaillé via Groq, analyse par sections, extraction de mots‑clés (TF‑IDF), détection de compétences, et export PDF.

## Fonctionnalités

- Upload de CV au format **PDF** et saisie de l’offre d’emploi
- Score **ATS** (similarité BERT / SentenceTransformers)
- Rapport IA (points notés /5, explications, suggestions d’amélioration)
- Analyse **section par section** (compétences, expérience, formation, projets, soft skills)
- Mots‑clés stratégiques (TF‑IDF) : présents dans l’offre, manquants dans le CV
- Détection de compétences “sémantique” (matching + score)
- Dashboard moderne (sidebar + onglets + status/toast)
- Export du rapport en **PDF**

## Démo

<img src="https://github.com/mamadouwhile/AI-Resume-Analyzer/tree/main/assets/image1" alt="screenshot" />
<img src="https://github.com/mamadouwhile/AI-Resume-Analyzer/tree/main/assets/image2" alt="screenshot" />
<img src="https://github.com/mamadouwhile/AI-Resume-Analyzer/tree/main/assets/image3" alt="screenshot" />
<img src="https://github.com/mamadouwhile/AI-Resume-Analyzer/tree/main/assets/image4" alt="screenshot" />
<img src="https://github.com/mamadouwhile/AI-Resume-Analyzer/tree/main/assets/image5" alt="screenshot" />


## Stack

- Streamlit (UI)
- pdfminer.six (extraction texte PDF)
- sentence-transformers + scikit-learn (embeddings + cosine similarity + TF‑IDF)
- Groq API (LLM)
- ReportLab (génération PDF)
- matplotlib (radar chart)

## Prérequis

- Python 3.10+ recommandé
- Une clé API Groq

## Installation

```bash
git clone https://github.com/<ton-user>/<ton-repo>.git
cd <ton-repo>

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
