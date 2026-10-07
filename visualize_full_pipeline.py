import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_hybrid_sanity_check(df):
    """Prints the highest and lowest scoring answers comparing Syntactic vs Semantic metrics."""
    print("\n" + "="*60)
    print("🕵️‍♂️ FULL HYBRID PIPELINE SANITY CHECK: EXTREME CASES")
    print("="*60)
    
    sorted_df = df.sort_values(by='final_grade_out_of_10', ascending=False)
    
    print("\n✅ TOP 2 HIGHEST SCORING ANSWERS:")
    for index, row in sorted_df.head(2).iterrows():
        print("-" * 50)
        student_txt = str(row.get('student_answer', row.get('student_clean', '')))[:110]
        print(f"Student Answer: {student_txt}...")
        print(f"Scores -> Syntactic: {row['midsem_algo_score']:.2f} | "
              f"Semantic: {row['semantic_score']:.2f} | "
              f"Final Grade: {row['final_grade_out_of_10']:.2f}/10")

    print("\n❌ TOP 2 LOWEST SCORING ANSWERS:")
    for index, row in sorted_df.tail(2).iterrows():
        print("-" * 50)
        student_txt = str(row.get('student_answer', row.get('student_clean', '')))[:110]
        print(f"Student Answer: {student_txt}...")
        print(f"Scores -> Syntactic: {row['midsem_algo_score']:.2f} | "
              f"Semantic: {row['semantic_score']:.2f} | "
              f"Final Grade: {row['final_grade_out_of_10']:.2f}/10")


def plot_full_grading_distributions(df):
    """Generates a 4-panel visual report tailored for your final submission PPT."""
    print("\n📊 Generating 4-panel comparison graphs...")
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    # 1. Bi-Gram Score Distribution
    sns.histplot(df['bigram_score'], bins=15, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('1. Bi-Gram Overlap (Syntactic)')
    axes[0].set_xlabel('Score (0 to 1)')
    axes[0].set_ylabel('Number of Students')
    
    # 2. Concept Coverage Distribution
    sns.histplot(df['concept_coverage'], bins=15, kde=True, ax=axes[1], color='lightgreen')
    axes[1].set_title('2. Concept Coverage (Chunking)')
    axes[1].set_xlabel('Score (0 to 1)')
    axes[1].set_ylabel('Number of Students')
    
    # 3. Semantic Similarity (Transformers)
    sns.histplot(df['semantic_score'], bins=15, kde=True, ax=axes[2], color='mediumpurple')
    axes[2].set_title('3. Semantic Similarity (Sentence-BERT)')
    axes[2].set_xlabel('Score (0 to 1)')
    axes[2].set_ylabel('Number of Students')
    
    # 4. Final Hybrid Grade (Out of 10)
    sns.histplot(df['final_grade_out_of_10'], bins=15, kde=True, ax=axes[3], color='salmon')
    axes[3].set_title('4. Final Hybrid Grade (Out of 10)')
    axes[3].set_xlabel('Grade (0 to 10)')
    axes[3].set_ylabel('Number of Students')
    
    plt.tight_layout()
    plt.savefig('final_hybrid_grading_visuals.png', dpi=300)
    print("✅ Graphs saved as 'final_hybrid_grading_visuals.png'. Ready to paste into PPT!")
    
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    input_file = "full_graded_dataset.csv"
    
    try:
        df = pd.read_csv(input_file)
        run_hybrid_sanity_check(df)
        plot_full_grading_distributions(df)
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{input_file}'. Ensure you ran 'semantic_grader.py' first!")