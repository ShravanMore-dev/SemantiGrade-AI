import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ocr_engine import OCREngine
from evaluator import AnswerEvaluator
from llm_feedback import LocalLLMFeedback

st.set_page_config(page_title="Automated Essay & Answer Evaluator", layout="wide")

# Force fresh initialization
evaluator = AnswerEvaluator()
llm = LocalLLMFeedback()

st.title("🎓 Hybrid NLP Answer Evaluation System")
st.caption("Syntactic Chunking + Sentence-BERT Semantic Scoring + Local LLM Feedback Engine")

# File Upload Section
st.header("1. Upload Documents (Scanned PDF/Images or Text)")
col1, col2 = st.columns(2)

with col1:
    ref_file = st.file_uploader("Upload Model / Reference Answer", type=['pdf', 'png', 'jpg', 'jpeg', 'txt'], key="ref_uploader")
    
with col2:
    student_files = st.file_uploader("Upload Student Submission(s)", type=['pdf', 'png', 'jpg', 'jpeg', 'txt'], accept_multiple_files=True, key="student_uploader")

if ref_file and student_files:
    if st.button("🚀 Process & Evaluate Submissions", type="primary"):
        with st.spinner("Executing OCR extraction and running full evaluation pipeline..."):
            
            # Step 1: Extract Model Answer (Fresh run)
            ref_bytes = ref_file.read()
            ref_text = OCREngine.extract_text(ref_bytes, ref_file.name)
            
            results = []
            
            # Step 2: Extract & Grade Student Submissions
            for s_file in student_files:
                s_bytes = s_file.read()
                s_text = OCREngine.extract_text(s_bytes, s_file.name)
                
                # Recompute metrics entirely in backend
                scores = evaluator.evaluate_submission(s_text, ref_text)
                
                # Fetch LLM Feedback
                feedback = llm.generate_feedback(ref_text, s_text, scores)
                
                results.append({
                    "Filename": s_file.name,
                    "Student Text": s_text,
                    "Concept Coverage": scores['concept_coverage'],
                    "Bi-Gram Overlap": scores['bigram_score'],
                    "Semantic Similarity": scores['semantic_score'],
                    "Grade (/10)": scores['final_grade_out_of_10'],
                    "LLM Feedback": feedback
                })

            df_results = pd.DataFrame(results)

            st.success("✅ Evaluation Complete!")
            
            # Show Model Answer
            with st.expander("📄 View Extracted Reference Model Answer"):
                st.write(ref_text)

            # Summary Table
            st.header("2. Graded Results Table")
            st.dataframe(df_results[["Filename", "Concept Coverage", "Bi-Gram Overlap", "Semantic Similarity", "Grade (/10)"]], use_container_width=True)

            # Distribution Analytics Graph
            st.header("3. Score Distributions Dashboard")
            sns.set_theme(style="whitegrid")
            fig, axes = plt.subplots(1, 4, figsize=(20, 4))
            
            sns.histplot(df_results['Bi-Gram Overlap'], bins=10, kde=True, ax=axes[0], color='skyblue')
            axes[0].set_title('Bi-Gram Overlap')
            
            sns.histplot(df_results['Concept Coverage'], bins=10, kde=True, ax=axes[1], color='lightgreen')
            axes[1].set_title('Concept Coverage')
            
            sns.histplot(df_results['Semantic Similarity'], bins=10, kde=True, ax=axes[2], color='mediumpurple')
            axes[2].set_title('Semantic Similarity')
            
            sns.histplot(df_results['Grade (/10)'], bins=10, kde=True, ax=axes[3], color='salmon')
            axes[3].set_title('Final Grade (/10)')
            
            st.pyplot(fig)

            # Detail Analysis & LLM Feedback
            st.header("4. Detailed Feedback & OCR Inspection")
            for idx, row in df_results.iterrows():
                with st.expander(f"📌 {row['Filename']} — Grade: {row['Grade (/10)']}/10"):
                    st.subheader("Extracted Student Text (OCR Result):")
                    st.info(row["Student Text"])
                    
                    st.subheader("🤖 Local LLM Evaluation & Suggestions:")
                    st.markdown(row["LLM Feedback"])