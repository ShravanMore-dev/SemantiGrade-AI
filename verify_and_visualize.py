import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_sanity_check(df):
    """Prints the highest and lowest scoring answers to verify the logic manually."""
    print("\n" + "="*50)
    print("🕵️‍♂️ SCRIPT SANITY CHECK: EXTREME CASES")
    print("="*50)
    
    # Sort dataframe by our combined midsem score
    sorted_df = df.sort_values(by='midsem_algo_score', ascending=False)
    
    print("\n✅ TOP 2 HIGHEST SCORING ANSWERS:")
    for index, row in sorted_df.head(2).iterrows():
        print("-" * 40)
        # Using your exact column names: model_clean and student_clean
        print(f"Model Answer: {str(row['model_clean'])[:120]}...")
        print(f"Student Answer: {str(row['student_clean'])[:120]}...")
        print(f"Scores -> Bi-gram: {row['bigram_score']:.2f} | Concept: {row['concept_coverage']:.2f} | Total: {row['midsem_algo_score']:.2f}")

    print("\n❌ TOP 2 LOWEST SCORING ANSWERS:")
    for index, row in sorted_df.tail(2).iterrows():
        print("-" * 40)
        print(f"Model Answer: {str(row['model_clean'])[:120]}...")
        print(f"Student Answer: {str(row['student_clean'])[:120]}...")
        print(f"Scores -> Bi-gram: {row['bigram_score']:.2f} | Concept: {row['concept_coverage']:.2f} | Total: {row['midsem_algo_score']:.2f}")


def plot_score_distributions(df):
    """Generates graphs for your midsem PPT slides."""
    print("\n📊 Generating graphs for your presentation...")
    
    # Set the style for nice looking academic plots
    sns.set_theme(style="whitegrid") #gging identifies nouns verbs adjectives
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Bigram Score Distribution
    sns.histplot(df['bigram_score'], bins=20, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution of Bi-Gram Overlap')
    axes[0].set_xlabel('Score (0 to 1)')
    axes[0].set_ylabel('Number of Students')
    
    # Plot 2: Concept Coverage Distribution
    sns.histplot(df['concept_coverage'], bins=20, kde=True, ax=axes[1], color='lightgreen')
    axes[1].set_title('Distribution of Concept Coverage (Chunking)')
    axes[1].set_xlabel('Score (0 to 1)')
    axes[1].set_ylabel('Number of Students')
    
    # Plot 3: Combined Midsem Algorithm Score
    sns.histplot(df['midsem_algo_score'], bins=20, kde=True, ax=axes[2], color='salmon')
    axes[2].set_title('Final Syntactic Grade Distribution')
    axes[2].set_xlabel('Score (0 to 1)')
    axes[2].set_ylabel('Number of Students')
    
    plt.tight_layout()
    
    # Save the plot so you can easily copy-paste into PPT
    plt.savefig('midsem_grading_visuals.png', dpi=300)
    print("✅ Graphs saved as 'midsem_grading_visuals.png' in your folder. Put these in your PPT!")
    
    # Attempt to open the window to show the plot (might not work over remote SSH, but saves regardless)
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    input_file = "midsem_syntactic_features.csv"
    
    try:
        df = pd.read_csv(input_file)
        run_sanity_check(df)
        plot_score_distributions(df)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}. Make sure you ran the previous script first!")