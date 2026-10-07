import sys
import types

# Linux python error workaround, not important
dummy_inisec = types.ModuleType("nltk.inisec")
dummy_inisec.find_spec = lambda *args, **kwargs: None
sys.modules["nltk.inisec"] = dummy_inisec

import pandas as pd
import re
import spacy
from nltk.stem import PorterStemmer

# Load spaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    sys.exit(1)

stemmer = PorterStemmer()
NEGATION_WORDS = {"not", "no", "never", "without", "n't"}

def preprocess_text(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {
            "clean_text": "",
            "tokens": [],
            "filtered_tokens": [],
            "lemmas": [],
            "stems": [],
            "noun_chunks": []
        }

    # 1. Regex Cleaning & Lowercasing
    clean_text = text.lower().strip()
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # 2. Tokenization & POS Analysis
    doc = nlp(clean_text)
    tokens = [token.text for token in doc]
    
    # 3. Stopword Filtering (Retaining Negations)
    filtered_tokens = [t.text for t in doc if (not t.is_stop or t.text in NEGATION_WORDS)]

    # 4. Lemmatization & Stemming
    lemmas = [t.lemma_ for t in doc if (not t.is_stop or t.text in NEGATION_WORDS)]
    stems = [stemmer.stem(w) for w in filtered_tokens]

    # 5. Noun Chunk Extraction (Exp 7)
    noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 0]

    return {
        "clean_text": clean_text,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "lemmas": lemmas,
        "stems": stems,
        "noun_chunks": noun_chunks
    }

def process_dataset(input_csv: str, output_csv: str):
    print(f"📖 Loading dataset from '{input_csv}'...")
    df = pd.read_csv(input_csv)

    model_col = "model_answer" if "model_answer" in df.columns else df.columns[1]
    student_col = "student_answer" if "student_answer" in df.columns else df.columns[2]

    print(f"🔄 Preprocessing Reference Answers ('{model_col}')...")
    model_results = df[model_col].apply(preprocess_text)

    print(f"🔄 Preprocessing Student Answers ('{student_col}')...")
    student_results = df[student_col].apply(preprocess_text)

    df["model_clean"] = [r["clean_text"] for r in model_results]
    df["model_lemmas"] = [" ".join(r["lemmas"]) for r in model_results]
    df["model_stems"] = [" ".join(r["stems"]) for r in model_results]
    df["model_noun_chunks"] = [" | ".join(r["noun_chunks"]) for r in model_results]

    df["student_clean"] = [r["clean_text"] for r in student_results]
    df["student_lemmas"] = [" ".join(r["lemmas"]) for r in student_results]
    df["student_stems"] = [" ".join(r["stems"]) for r in student_results]
    df["student_noun_chunks"] = [" | ".join(r["noun_chunks"]) for r in student_results]

    df.to_csv(output_csv, index=False)
    print(f"\n✅ Preprocessing complete! Exported to '{output_csv}'.")

if __name__ == "__main__":
    process_dataset("dataset_fixed.csv", "exp1_preprocessed_output.csv")