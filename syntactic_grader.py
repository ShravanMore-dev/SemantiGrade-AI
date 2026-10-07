import pandas as pd
import nltk
from nltk.util import ngrams
from nltk.chunk import RegexpParser
from nltk.tokenize import word_tokenize

# Download required NLTK resources silently
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # <--- WE ADDED THIS NEW REQUIREMENT
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True) # <--- Added this too just in case NLTK 3.10 gets mad about the tagger next!

def calculate_ngram_similarity(student_text, model_text, n=2):
    """
    Calculates Jaccard Similarity between n-grams of two texts.
    (Syllabus Exp 4 Integration)
    """
    if not isinstance(student_text, str) or not isinstance(model_text, str):
        return 0.0
        
    student_tokens = word_tokenize(student_text)
    model_tokens = word_tokenize(model_text)
    
    if len(student_tokens) < n or len(model_tokens) < n:
        return 0.0
        
    student_ngrams = set(ngrams(student_tokens, n))
    model_ngrams = set(ngrams(model_tokens, n))
    
    intersection = len(student_ngrams.intersection(model_ngrams))
    union = len(student_ngrams.union(model_ngrams))
    
    return intersection / union if union > 0 else 0.0

def extract_technical_chunks(text):
    """
    Extracts Noun Phrases (NP) to identify key concepts.
    (Syllabus Exp 7 Integration)
    """
    if not isinstance(text, str):
        return []
        
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    
    chunk_grammar = "NP: {<DT>?<JJ>*<NN|NNS|NNP|NNPS>+}"
    chunk_parser = RegexpParser(chunk_grammar)
    
    tree = chunk_parser.parse(tagged)
    
    noun_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            phrase = " ".join([word for word, tag in subtree.leaves()])
            noun_phrases.append(phrase.lower())
            
    return noun_phrases

def concept_coverage_score(student_text, model_text):
    """
    Checks what percentage of model answer concepts exist in the student answer.
    """
    model_concepts = set(extract_technical_chunks(model_text))
    student_concepts = set(extract_technical_chunks(student_text))
    
    if not model_concepts:
        return 0.0
        
    matched_concepts = model_concepts.intersection(student_concepts)
    return len(matched_concepts) / len(model_concepts)

if __name__ == "__main__":
    print("🚀 Firing up the Syntactic Grader (Exp 4 & 7)...")
    
    input_file = "exp1_preprocessed_output.csv"
    
    try:
        df = pd.read_csv(input_file)
        
        # Mapping to your actual CSV column names: 'student_clean' & 'model_clean'
        print("Calculating Bi-gram and Tri-gram overlaps...")
        df['bigram_score'] = df.apply(
            lambda row: calculate_ngram_similarity(str(row['student_clean']), str(row['model_clean']), n=2), 
            axis=1
        )
        df['trigram_score'] = df.apply(
            lambda row: calculate_ngram_similarity(str(row['student_clean']), str(row['model_clean']), n=3), 
            axis=1
        )
        
        print("Extracting Noun Phrases and calculating concept coverage...")
        df['concept_coverage'] = df.apply(
            lambda row: concept_coverage_score(str(row['student_clean']), str(row['model_clean'])), 
            axis=1
        )
        
        # Combined syntactic score for midsem presentation
        df['midsem_algo_score'] = (df['bigram_score'] + df['concept_coverage']) / 2
        
        output_file = "midsem_syntactic_features.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Success, bhidu! Extracted features saved to {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}. Make sure it's in the same folder!")