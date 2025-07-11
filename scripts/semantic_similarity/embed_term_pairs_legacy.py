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
    The main event! Now speaking the universal language of data!
    """
    parser = argparse.ArgumentParser(description="Embed term pairs using classic models like Word2Vec or GloVe.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model file.")
    parser.add_argument("--term_pairs_file", type=str, required=True, help="Path to the JSON file with term pairs.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to write the output CSV.")
    # --- HERE'S A NEW PART! It needs to know its own name! ---
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model being used.")
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
        # --- A NEW, PERFECT HEADER! ---
        writer.writerow(['term', 'counterpart', 'cosine_similarity', 'model'])

        print("🧠 Let's start thinking! Calculating similarities...")
        for pair in term_pairs:
            term = pair['term']
            counterpart = pair['counterpart']
            
            term_words = term.lower().split()
            counterpart_words = counterpart.lower().split()
            
            vec1 = get_avg_vector(term_words, model).reshape(1, -1)
            vec2 = get_avg_vector(counterpart_words, model).reshape(1, -1)
            
            similarity = cosine_similarity(vec1, vec2)[0][0]
            
            # --- AND A NEW, PERFECT ROW WITH THE MODEL'S NAME! ---
            writer.writerow([term, counterpart, similarity, args.model_name])

    print(f"🎉 All done! Results are saved in {args.output_file}!")

if __name__ == "__main__":
    main()