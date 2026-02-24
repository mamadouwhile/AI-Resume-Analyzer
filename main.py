# app.py
import os
import io
import re

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from pdfminer.high_level import extract_text

from dotenv import load_dotenv
from groq import Groq

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


# ------------------ PAGE CONFIG ------------------

st.set_page_config(page_title="Analyseur Intelligent de CV", page_icon="📊", layout="wide")
st.title("Analyseur Intelligent de CV")


# ------------------ ENV ------------------

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


# ------------------ SESSION STATE ------------------

st.session_state.setdefault("resume", "")
st.session_state.setdefault("job_desc", "")
st.session_state.setdefault("ran", False)

st.session_state.setdefault("ats_score", None)
st.session_state.setdefault("report", "")
st.session_state.setdefault("section_report", "")
st.session_state.setdefault("keywords", [])
st.session_state.setdefault("missing_keywords", [])
st.session_state.setdefault("matched_skills", [])
st.session_state.setdefault("missing_skills", [])
st.session_state.setdefault("avg_score", 0.0)
st.session_state.setdefault("skill_ratio", 0.0)


# ------------------ CACHES ------------------

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


@st.cache_resource
def get_groq_client():
    if not api_key:
        raise RuntimeError(
            "Clé GROQ manquante. Mets GROQ_API_KEY dans ton .env (ou variable d'environnement)."
        )
    return Groq(api_key=api_key)


# ------------------ CORE FUNCTIONS ------------------

def extract_pdf_text(uploaded_file) -> str:
    try:
        return extract_text(uploaded_file)
    except Exception as e:
        st.error(f"Erreur extraction PDF : {e}")
        return ""


def calculate_similarity_bert(text1: str, text2: str) -> float:
    model = get_embedding_model()
    e1 = model.encode([text1])
    e2 = model.encode([text2])
    return float(cosine_similarity(e1, e2)[0][0])


def get_report(resume: str, job_desc: str, model_name: str) -> str:
    client = get_groq_client()
    prompt = f"""
# CONTEXTE :
Vous êtes un expert en recrutement et analyse de CV.
Vous devez analyser un CV par rapport à une offre d'emploi.

# INSTRUCTIONS :
- Analysez tous les critères présents dans l'offre (compétences, expérience, diplômes, etc.).
- Attribuez une note sur 5 pour chaque critère (exemple : 3/5).
- Commencez chaque point par la note + emoji :
    ✅ = Correspondance forte
    ❌ = Ne correspond pas
    ⚠️ = Partiellement ou incertain
- Donnez une explication détaillée pour chaque point.
- Terminez par une section intitulée : "Suggestions d'amélioration du CV :"

# DONNÉES :
CV :
{resume}

---
Offre :
{job_desc}

# FORMAT :
Note/5 + Emoji + Explication détaillée
"""
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
    )
    return chat_completion.choices[0].message.content


def get_section_analysis(resume: str, job_desc: str, model_name: str) -> str:
    client = get_groq_client()
    prompt = f"""
Analyse le CV section par section par rapport à l'offre d'emploi.

Sections :
1. Compétences techniques
2. Expérience professionnelle
3. Formation
4. Projets
5. Soft skills

Pour chaque section :
- Donne une note sur 5
- Explique en détail
- Indique les points forts
- Indique les axes d'amélioration

CV :
{resume}

Offre :
{job_desc}
"""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
    )
    return response.choices[0].message.content


def extract_scores(text: str) -> list[float]:
    matches = re.findall(r"(\d+(?:\.\d+)?)/5", text)
    return [float(m) for m in matches]


def generate_pdf(report_text: str) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    style = getSampleStyleSheet()["Normal"]

    for line in report_text.split("\n"):
        if line.strip():
            elements.append(Paragraph(line, style))
            elements.append(Spacer(1, 0.15 * inch))
        else:
            elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer


def detect_missing_keywords(resume: str, job_desc: str, top_k: int):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_k)
    _ = vectorizer.fit_transform([job_desc])
    keywords = list(vectorizer.get_feature_names_out())

    resume_lower = resume.lower()
    missing = [w for w in keywords if w not in resume_lower]
    return keywords, missing


def detect_skills_semantic(resume_text: str, threshold: float):
    model = get_embedding_model()
    skills_database = [
        "Python programming",
        "Object Oriented Programming",
        "SQL databases",
        "Web development",
        "Frontend development",
        "Backend development",
        "Machine Learning",
        "Cloud computing",
        "DevOps",
        "API development",
        "Test automation",
        "Data analysis",
    ]

    resume_embedding = model.encode([resume_text])
    matched, missing = [], []

    for skill in skills_database:
        s_emb = model.encode([skill])
        sim = float(cosine_similarity(s_emb, resume_embedding)[0][0])
        if sim >= threshold:
            matched.append((skill, round(sim, 2)))
        else:
            missing.append(skill)

    return matched, missing


def plot_radar(ats_score: float, avg_score: float, skill_ratio: float):
    labels = ["ATS", "IA", "Compétences"]
    values = [round(ats_score, 2), round(avg_score, 2), round(skill_ratio, 2)]
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.plot(angles, values, linewidth=3)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(color="#444", linestyle="dotted", linewidth=1)
    return fig


# ------------------ SIDEBAR (INPUTS) ------------------

with st.sidebar:
    st.header("Entrées")

    resume_file = st.file_uploader("CV (PDF)", type="pdf")
    job_desc = st.text_area("Offre d’emploi", height=200, placeholder="Colle ici la description du poste...")

    st.divider()
    st.subheader("Réglages")

    model_name = st.selectbox(
        "Modèle Groq",
        options=["llama-3.3-70b-versatile"],
        index=0,
    )

    skill_threshold = st.slider("Seuil similarité compétences", 0.10, 0.70, 0.35, 0.01)
    top_keywords = st.slider("Nombre de mots-clés (TF-IDF)", 5, 40, 20, 1)

    st.divider()
    run = st.button("Lancer l’analyse", type="primary")
    reset = st.button("Nouvelle analyse")

    if reset:
        for k in list(st.session_state.keys()):
            if k in {
                "resume", "job_desc", "ran",
                "ats_score", "report", "section_report",
                "keywords", "missing_keywords",
                "matched_skills", "missing_skills",
                "avg_score", "skill_ratio",
            }:
                st.session_state[k] = "" if isinstance(st.session_state[k], str) else None
        st.session_state.ran = False
        st.rerun()


# ------------------ MAIN WORKFLOW ------------------

if run:
    if resume_file is None or not job_desc.strip():
        st.warning("Ajoute un CV PDF + colle l’offre d’emploi.")
        st.stop()

    st.session_state.resume = extract_pdf_text(resume_file)
    st.session_state.job_desc = job_desc

    with st.status("Analyse en cours…", expanded=True, state="running") as status:
        st.write("1) Calcul ATS…")
        ats = calculate_similarity_bert(st.session_state.resume, st.session_state.job_desc)
        st.session_state.ats_score = ats

        st.write("2) Génération du rapport IA…")
        rep = get_report(st.session_state.resume, st.session_state.job_desc, model_name)
        st.session_state.report = rep

        st.write("3) Analyse par sections…")
        sec = get_section_analysis(st.session_state.resume, st.session_state.job_desc, model_name)
        st.session_state.section_report = sec

        st.write("4) Extraction scores & mots-clés…")
        scores = extract_scores(rep)
        st.session_state.avg_score = (sum(scores) / (5 * len(scores))) if scores else 0.0

        kw, miss_kw = detect_missing_keywords(st.session_state.resume, st.session_state.job_desc, top_keywords)
        st.session_state.keywords = kw
        st.session_state.missing_keywords = miss_kw

        st.write("5) Compétences sémantiques…")
        matched, missing = detect_skills_semantic(st.session_state.resume, skill_threshold)
        st.session_state.matched_skills = matched
        st.session_state.missing_skills = missing

        if matched or missing:
            st.session_state.skill_ratio = len(matched) / (len(matched) + len(missing))
        else:
            st.session_state.skill_ratio = 0.0

        status.update(label="Analyse terminée", state="complete", expanded=False)  # status UI [web:43]

    st.toast("Analyse terminée ✅")  # toast UI [web:40]
    st.session_state.ran = True


# ------------------ DISPLAY ------------------

if not st.session_state.ran:
    st.info("Renseigne les entrées à gauche puis clique “Lancer l’analyse”.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Dashboard", "Rapport", "Sections", "Mots-clés", "Compétences"]
)

with tab1:
    c1, c2, c3 = st.columns(3, border=True)
    c1.metric("ATS", f"{round((st.session_state.ats_score or 0) * 100, 2)} %")
    c2.metric("IA (moyenne)", f"{round(st.session_state.avg_score * 100, 2)} %")
    c3.metric("Couverture compétences", f"{round(st.session_state.skill_ratio * 100, 2)} %")

    st.subheader("Radar")
    fig = plot_radar(
        st.session_state.ats_score or 0.0,
        st.session_state.avg_score,
        st.session_state.skill_ratio,
    )
    st.pyplot(fig)

with tab2:
    st.subheader("Rapport IA")
    st.markdown(
        f"<div style='background-color:#0f1117; padding:14px; border-radius:12px;'>{st.session_state.report}</div>",
        unsafe_allow_html=True,
    )
    pdf = generate_pdf(st.session_state.report)
    st.download_button(
        "Télécharger le rapport (PDF)",
        data=pdf,
        file_name="rapport_analyse_cv.pdf",
        mime="application/pdf",
    )

with tab3:
    st.subheader("Analyse par section")
    st.markdown(st.session_state.section_report)

with tab4:
    colA, colB = st.columns(2)
    with colA:
        st.info("Mots-clés détectés (offre)")
        for w in st.session_state.keywords:
            st.write(f"• {w}")
    with colB:
        st.warning("Manquants (CV)")
        for w in st.session_state.missing_keywords:
            st.write(f"❌ {w}")

with tab5:
    colA, colB = st.columns(2)
    with colA:
        st.success("Compétences détectées")
        if st.session_state.matched_skills:
            for skill, score in st.session_state.matched_skills:
                st.write(f"✅ {skill} (sim: {score})")
        else:
            st.write("Aucune au-dessus du seuil.")
    with colB:
        st.error("Faibles/absentes")
        if st.session_state.missing_skills:
            for skill in st.session_state.missing_skills:
                st.write(f"❌ {skill}")
        else:
            st.write("Tout est couvert.")
