## 📄 Resume Analyzer (ATS-Style)

This project is an ATS-style resume analysis web application designed to evaluate how well a candidate’s resume aligns with a given job description. It leverages Natural Language Processing (NLP) and machine learning techniques to compute a realistic resume–job match score and provide actionable keyword insights.

The system processes resumes in PDF format, performs text extraction and preprocessing, and represents both the resume and job description using TF-IDF vectorization with uni-grams and bi-grams. Cosine similarity is used to measure textual similarity, followed by score normalization and skill-based boosting to better approximate real-world ATS scoring behavior.

In addition to scoring, the application extracts important keywords from both the resume and the job description using part-of-speech tagging, and identifies missing or underrepresented skills that can be added to improve resume relevance. The entire pipeline is wrapped in an interactive Streamlit interface for ease of use.

### 🔧 Technical Highlights
- PDF text extraction using **PyPDF2**
- Text preprocessing: lowercasing, regex-based cleaning, tokenization
- NLP processing with **NLTK** (tokenization, stopwords, POS tagging)
- Feature representation using **TF-IDF (1–2 grams)**
- Similarity computation via **Cosine Similarity**
- ATS-style score normalization for realistic match percentages
- Skill overlap–based score boosting to simulate recruiter weighting
- Keyword frequency analysis and missing skill detection
- Interactive web UI built with **Streamlit**

### 🛠 Tech Stack
- **Programming Language:** Python  
- **Libraries & Tools:** Streamlit, Scikit-learn, NLTK, PyPDF2, Matplotlib  
- **Techniques:** NLP, TF-IDF Vectorization, Cosine Similarity, Keyword Extraction  

This project is intended for learning and demonstration purposes and provides a practical understanding of how basic Applicant Tracking Systems evaluate resumes using NLP-driven similarity matching.
