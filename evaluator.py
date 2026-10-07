import re
import nltk
from nltk.util import ngrams
import torch
from sentence_transformers import SentenceTransformer, util

# Ensure NLTK data is ready
for resource in ['punkt', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

class AnswerEvaluator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # CPU/GPU routing fallback
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception:
            self.device = "cpu"
            self.model = SentenceTransformer(model_name, device="cpu")

    def _get_bigrams(self, text: str) -> set:
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(ngrams(tokens, 2))

    def compute_bigram_overlap(self, student_text: str, reference_text: str) -> float:
        student_bg = self._get_bigrams(student_text)
        ref_bg = self._get_bigrams(reference_text)
        if not ref_bg:
            return 0.0
        return len(student_bg.intersection(ref_bg)) / len(ref_bg)

    def compute_concept_coverage(self, student_text: str, reference_text: str) -> float:
        ref_tokens = nltk.word_tokenize(reference_text.lower())
        ref_pos = nltk.pos_tag(ref_tokens)
        
        # Extract Noun/Verb concept keywords from reference answer
        concepts = {word for word, pos in ref_pos if pos.startswith(('NN', 'VB')) and len(word) > 2}
        if not concepts:
            return 0.0
            
        student_tokens = set(nltk.word_tokenize(student_text.lower()))
        matched = concepts.intersection(student_tokens)
        return len(matched) / len(concepts)

    def compute_semantic_similarity(self, student_text: str, reference_text: str) -> float:
        try:
            emb_student = self.model.encode(student_text, convert_to_tensor=True, device=self.device)
            emb_ref = self.model.encode(reference_text, convert_to_tensor=True, device=self.device)
            similarity = util.cos_sim(emb_student, emb_ref).item()
            return max(0.0, float(similarity))
        except Exception:
            # CPU Fallback if HIP/CUDA dispatches error at runtime
            emb_student = self.model.encode(student_text, convert_to_tensor=True, device="cpu")
            emb_ref = self.model.encode(reference_text, convert_to_tensor=True, device="cpu")
            return max(0.0, float(util.cos_sim(emb_student, emb_ref).item()))

    def evaluate_submission(self, student_text: str, reference_text: str) -> dict:
        bigram_score = self.compute_bigram_overlap(student_text, reference_text)
        concept_score = self.compute_concept_coverage(student_text, reference_text)
        semantic_score = self.compute_semantic_similarity(student_text, reference_text)

        # UPDATED WEIGHTS: 60% Semantic, 25% Concept Coverage, 15% Bi-Gram
        weighted_score = (
            (0.60 * semantic_score) + 
            (0.25 * concept_score) + 
            (0.15 * bigram_score)
        )
        final_grade = round(weighted_score * 10, 2)

        return {
            "bigram_score": round(bigram_score, 4),
            "concept_coverage": round(concept_score, 4),
            "semantic_score": round(semantic_score, 4),
            "final_grade_out_of_10": final_grade
        }