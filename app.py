import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import PyPDF2
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ---------------- NLTK DOWNLOADS ----------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Resume Analyzer for Job Role",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Analyzer")
st.markdown("""
Upload your **resume (PDF)** and paste a **job description** to evaluate how well your resume matches the role.  
This tool uses **TF-IDF + Cosine Similarity** (ATS-style matching).
""")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("About")
    st.info("""
    ✔ Resume–Job match score  
    ✔ Important skill keywords  
    ✔ Missing skills suggestion  
    ✔ ATS-style matching logic  
    """)
    st.header("How it works")
    st.write("""
    1. Upload resume (PDF)
    2. Paste job description
    3. Click Analyze Resume
    4. Improve resume using missing keywords
    """)

# ---------------- HELPERS ----------------
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"PDF Error: {e}")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    keep_words = {"experience", "skills", "knowledge", "using", "worked"}
    words = word_tokenize(text)
    return " ".join([w for w in words if w not in stop_words or w in keep_words])

def skill_overlap_boost(resume_text, job_text):
    skill_list = [
        "python", "machine learning", "deep learning", "nlp", "llm",
        "rag", "faiss", "vector database", "streamlit", "docker",
        "sql", "api", "langchain", "pytorch", "tensorflow"
    ]
    resume_text = resume_text.lower()
    job_text = job_text.lower()
    matched = sum(1 for skill in skill_list if skill in resume_text and skill in job_text)
    return min(matched * 3, 20)  # max 20% boost

def calculate_similarity(resume_text, job_text):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_text))
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vectorizer.fit_transform([resume_processed, job_processed])
    # --- RAW COSINE SIMILARITY ---
    raw_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    # --- NORMALIZATION (ATS STYLE) ---
    normalized_score = min(100, (raw_score / 0.35) * 100)
    # --- SKILL BOOST ---
    boost = skill_overlap_boost(resume_processed, job_processed)
    # --- FINAL SCORE ---
    final_score = min(100, normalized_score + boost)
    return round(final_score, 2), resume_processed, job_processed, vectorizer.get_feature_names_out()

def extract_keywords(text, top_n=15):
    words = word_tokenize(text)
    tagged = pos_tag(words)
    keywords = [w for w, pos in tagged if pos.startswith("NN") or pos.startswith("JJ")]
    return Counter(keywords).most_common(top_n)

# ---------------- MAIN APP ----------------
def main():
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_description = st.text_area("Paste Job Description", height=220)

    if st.button("Analyze Resume"):
        if not uploaded_file:
            st.warning("Please upload your resume")
            return
        if not job_description:
            st.warning("Please paste the job description")
            return

        with st.spinner("Analyzing Resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text.strip():
                st.error("Unable to extract text from PDF.")
                return

            score, resume_processed, job_processed, features = calculate_similarity(
                resume_text, job_description
            )

            # ---------------- RESULTS ----------------
            st.subheader("📊 Resume Analysis Result")
            st.metric("Resume Match Score", f"{score}%")

            fig, ax = plt.subplots(figsize=(7, 1))
            color = "#ff4b4b" if score < 40 else "#ffa726" if score < 70 else "#0f9d58"
            ax.barh([0], [score], color=color)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel("Match Percentage")
            st.pyplot(fig)

            if score < 40:
                st.warning("Low match — Resume needs improvement.")
            elif score < 70:
                st.info("Good match — Minor improvements recommended.")
            else:
                st.success("Excellent match — Resume fits the role well.")

            # ---------------- KEYWORDS ----------------
            st.subheader("🔑 Job Description Keywords")
            job_keywords = extract_keywords(job_processed)
            st.write([kw for kw, _ in job_keywords])

            st.subheader("📄 Resume Keywords")
            resume_keywords = extract_keywords(resume_processed)
            st.write([kw for kw, _ in resume_keywords])

            # ---------------- MISSING SKILLS ----------------
            job_words = set([kw for kw, _ in job_keywords])
            resume_words = set([kw for kw, _ in resume_keywords])
            missing = sorted(job_words - resume_words)

            st.subheader("❌ Missing Important Keywords")
            if missing:
                st.error(", ".join(missing))
            else:
                st.success("Your resume covers all major keywords 🎯")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()