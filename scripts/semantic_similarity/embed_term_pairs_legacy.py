import os
import json
import csv
import argparse
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

def get_avg_vector(terms, model):
    """
    Averages the vectors for a list of words.
    Ignores words that aren't in the vocabulary! So smart!
    """
    vectors = [model[word] for word in terms if word in model.key_to_index]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def main():
    """
    The main event! Now with the right keys to the treasure!
    """
    parser = argparse.ArgumentParser(description="Embed term pairs using classic models like Word2Vec or GloVe.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model file.")
    parser.add_argument("--term_pairs_file", type=str, required=True, help="Path to the JSON file with term pairs.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to write the output CSV.")
    parser.add_argument("--is_word2vec", action='store_true', help="Flag if the model is in Word2Vec binary format.")
    args = parser.parse_args()

    print(f"📦 Loading model from: {args.model_path}")
    if args.is_word2vec:
        model = KeyedVectors.load_word2vec_format(args.model_path, binary=True)
    else:
        model = KeyedVectors.load_word2vec_format(args.model_path, binary=False, no_header=True)
    print("✅ Model loaded!")

    print(f"📚 Reading term pairs from your beautiful JSON file: {args.term_pairs_file}")
    with open(args.term_pairs_file, 'r', encoding='utf-8') as infile:
        term_pairs = json.load(infile)

    with open(args.output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['term1', 'term2', 'similarity'])

        print("🧠 Let's start thinking! Calculating similarities...")
        # --- THIS IS THE NEW PART! ---
        for pair in term_pairs:
            # We use the right names to get our terms!
            term1_str = pair['term']
            term2_str = pair['counterpart']
            
            term1_words = term1_str.lower().split()
            term2_words = term2_str.lower().split()
            
            vec1 = get_avg_vector(term1_words, model).reshape(1, -1)
            vec2 = get_avg_vector(term2_words, model).reshape(1, -1)
            
            similarity = cosine_similarity(vec1, vec2)[0][0]
            
            writer.writerow([term1_str, term2_str, similarity])

    print(f"🎉 All done! Results are saved in {args.output_file}!")

if __name__ == "__main__":
    main()