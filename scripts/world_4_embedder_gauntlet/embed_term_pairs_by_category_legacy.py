# scripts/world_4_embedder_gauntlet/embed_term_pairs_legacy_by_category.py (NEW!)
import os
import json
import csv
import argparse
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def get_avg_vector(terms, model):
    vectors = [model[word] for word in terms if word in model.key_to_index]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model file.")
    parser.add_argument("--term_pairs_file", type=str, required=True, help="Path to the JSON file with categorized term pairs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the base output directory.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model being used.")
    parser.add_argument("--is_word2vec", action='store_true', help="Flag if the model is in Word2Vec binary format.")
    args = parser.parse_args()

    print(f"📦 Loading legacy contender: {args.model_name}")
    if args.is_word2vec:
        model = KeyedVectors.load_word2vec_format(args.model_path, binary=True)
    else:
        model = KeyedVectors.load_word2vec_format(args.model_path, binary=False, no_header=True)
    print("✅ Legacy model loaded!")

    with open(args.term_pairs_file, 'r', encoding='utf-8') as infile:
        all_pairs = json.load(infile)

    categorized_pairs = defaultdict(list)
    for pair in all_pairs:
        categorized_pairs[pair['category']].append(pair)

    for category, pairs in categorized_pairs.items():
        print(f"\n--- 🏟️  Starting legacy '{category}' tournament! ({len(pairs)} pairs) 🏟️  ---")
        
        category_output_dir = os.path.join(args.output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)
        output_path = os.path.join(category_output_dir, args.model_name + ".csv")

        with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['term', 'counterpart', 'cosine_similarity', 'model'])

            for pair in pairs:
                term = pair['term']
                counterpart = pair['counterpart']
                vec1 = get_avg_vector(term.lower().split(), model).reshape(1, -1)
                vec2 = get_avg_vector(counterpart.lower().split(), model).reshape(1, -1)
                similarity = cosine_similarity(vec1, vec2)[0][0]
                writer.writerow([term, counterpart, similarity, args.model_name])
        
        print(f"✅ Legacy '{category}' tournament complete! Results saved to {output_path}")

    print(f"\n🎉 All legacy tournaments for {args.model_name} are done! YAY!")

if __name__ == "__main__":
    main()