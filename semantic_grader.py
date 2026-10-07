import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

def compute_semantic_scores(input_csv, output_csv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Target Device: {device.upper()}")

    print("🚀 Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    df = pd.read_csv(input_csv)
    print(f"📊 Loaded {len(df)} records from {input_csv}")

    student_texts = df['student_answer'].fillna("").astype(str).tolist()
    model_texts = df['reference_answer'].fillna("").astype(str).tolist()

    print("🧠 Generating embeddings and computing Cosine Similarity...")
    try:
        student_embeddings = model.encode(student_texts, convert_to_tensor=True, device=device)
        model_embeddings = model.encode(model_texts, convert_to_tensor=True, device=device)
    except Exception as e:
        print(f"⚠️ GPU Kernel error: {e}")
        print("🔄 Falling back to CPU mode...")
        model = model.to("cpu")
        student_embeddings = model.encode(student_texts, convert_to_tensor=True, device="cpu")
        model_embeddings = model.encode(model_texts, convert_to_tensor=True, device="cpu")

    semantic_scores = []
    for i in range(len(df)):
        sim = util.cos_sim(student_embeddings[i], model_embeddings[i]).item()
        semantic_scores.append(max(0.0, float(sim)))

    df['semantic_score'] = semantic_scores

    # Calculate Hybrid Score: 40% Concept Coverage + 20% Bigram + 40% Semantic Similarity
    df['final_weighted_score'] = (
        (0.25 * df['concept_coverage']) + 
        (0.15 * df['bigram_score']) + 
        (0.60 * df['semantic_score'])
    )

    df['final_grade_out_of_10'] = (df['final_weighted_score'] * 10).round(2)
    df.to_csv(output_csv, index=False)
    print(f"✅ Semantic grading complete! Output saved to '{output_csv}'")

if __name__ == "__main__":
    compute_semantic_scores('midsem_syntactic_features.csv', 'full_graded_dataset.csv')