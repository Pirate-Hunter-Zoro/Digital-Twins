# scripts/world_4_embedder_gauntlet/analyze_term_pairs.py (SIMPLIFIED!)
import json
from pathlib import Path
import random
from collections import defaultdict

def analyze_term_pairs_by_category_and_type():
    """
    Analyzes our perfectly generated term pairs, breaking them down by
    category and source type to find interesting examples!
    """
    print("🤖✨ Welcome to the Simplified Salient Pair Analyzer! ✨🤖")

    data_file = Path("data/term_pairs_final_clean.json")

    if not data_file.exists():
        print(f"❌ OH NO! I can't find the data file at: {data_file}")
        return

    print(f"✅ Found the data at {data_file}! Let the analysis... BEGIN!")
    with open(data_file, 'r') as f:
        all_pairs = json.load(f)

    # --- Sort pairs by category AND then by type! ---
    categorized_pairs = defaultdict(lambda: defaultdict(list))
    for pair in all_pairs:
        categorized_pairs[pair['category']][pair['type']].append(pair)

    # --- Analyze each bucket! ---
    for category, types in categorized_pairs.items():
        print(f"\n\n--- 🔬 Analyzing Category: {category.upper()} 🔬 ---")
        
        for pair_type, pairs in types.items():
            print(f"\n  ---Analyzing Pair Type: '{pair_type}'---")

            print(f"  Total pairs found: {len(pairs)}")

            print(f"\n  --- ✨ Salient Examples for {category.title()} ({pair_type.replace('_', ' ').title()}) ✨ ---")
            if pairs:
                num_examples = min(5, len(pairs))
                random_examples = random.sample(pairs, num_examples)
                for i, pair in enumerate(random_examples):
                    print(f"    Example {i+1}:")
                    print(f"      Term       : {pair['term']}")
                    print(f"      Counterpart: {pair['counterpart']}")
            else:
                print("    No pairs were found in this sub-category!")

    print("\n\n🎉 AHAHAHA! The simplified analysis is complete!")

if __name__ == "__main__":
    analyze_term_pairs_by_category_and_type()