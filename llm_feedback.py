import requests
import json

class LocalLLMFeedback:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model: str = "llama3"):
        self.ollama_url = ollama_url
        self.model = model

    def generate_feedback(self, reference_text: str, student_text: str, scores: dict) -> str:
        prompt = f"""
You are an expert academic evaluator. Analyze the following student answer against the model answer.

MODEL ANSWER:
"{reference_text}"

STUDENT ANSWER:
"{student_text}"

EVALUATION SCORES:
- Concept Coverage: {scores['concept_coverage'] * 100:.1f}%
- Semantic Similarity: {scores['semantic_score'] * 100:.1f}%
- Final Grade: {scores['final_grade_out_of_10']}/10

Provide concise, highly specific feedback formatted as follows:
1. Key Missing Concepts (bullet points)
2. What the Student did well
3. Concrete Suggestions for improvement to reach 10/10

Be direct and precise. Maximum 150 words.
"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json().get("response", "Feedback generation succeeded but payload was empty.")
            else:
                return f"⚠️ Ollama local LLM returned status code {response.status_code}. Ensure model '{self.model}' is installed (`ollama run {self.model}`)."
        except requests.exceptions.RequestException:
            return "⚠️ Could not connect to local Ollama server. Make sure Ollama is running (`ollama serve`). Evaluation metrics above remain 100% accurate."