import os
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
    # Get all the vectors for the words we can find!
    vectors = [model[word] for word in terms if word in model.key_to_index]
    if not vectors:
        # If we can't find ANY words, we return a vector of zeros!
        return np.zeros(model.vector_size)
    # Average them all together!
    return np.mean(vectors, axis=0)

def main():
    """
    The main event! This is where the magic happens!
    """
    parser = argparse.ArgumentParser(description="Embed term pairs using classic models like Word2Vec or GloVe.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model file.")
    parser.add_argument("--term_pairs_file", type=str, required=True, help="Path to the CSV with term pairs.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to write the output CSV.")
    parser.add_argument("--is_word2vec", action='store_true', help="Flag if the model is in Word2Vec binary format.")
    args = parser.parse_args()

    print(f"📦 Loading model from: {args.model_path}")
    # We need to load Word2Vec a little differently!
    if args.is_word2vec:
        model = KeyedVectors.load_word2vec_format(args.model_path, binary=True)
    else:
        model = KeyedVectors.load_word2vec_format(args.model_path, binary=False, no_header=True)
    print("✅ Model loaded!")

    # Let's get our term pairs ready!
    with open(args.term_pairs_file, 'r', newline='', encoding='utf-8') as infile, \
         open(args.output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        # Write the header for our beautiful new results file!
        writer.writerow(['term1', 'term2', 'similarity'])
        # Skip the header from the input file
        next(reader)

        print("🧠 Let's start thinking! Calculating similarities...")
        for row in reader:
            term1_str, term2_str = row[0], row[1]
            
            # Split the terms into tiny little words!
            term1_words = term1_str.lower().split()
            term2_words = term2_str.lower().split()
            
            # Get our beautiful average vectors!
            vec1 = get_avg_vector(term1_words, model).reshape(1, -1)
            vec2 = get_avg_vector(term2_words, model).reshape(1, -1)
            
            # Calculate the cosine similarity!
            similarity = cosine_similarity(vec1, vec2)[0][0]
            
            writer.writerow([term1_str, term2_str, similarity])

    print(f"🎉 All done! Results are saved in {args.output_file}!")

if __name__ == "__main__":
    main()