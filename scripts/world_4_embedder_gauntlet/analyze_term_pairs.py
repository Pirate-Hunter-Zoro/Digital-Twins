import json
from pathlib import Path
import random
from collections import defaultdict

def analyze_term_pairs_by_category_and_type():
    """
    Analyzes term pairs, breaking them down first by category (diagnosis,
    procedure, medication) and then by their source type (code-based vs.
    LLM-generated). It's the ultimate analysis machine! AHAHAHA!
    """
    print("🤖✨ Welcome to the Categorical & Source Analyzer 7000! ✨🤖")

    # Let's find our fully generated data!
    data_file = Path("data/term_pairs_fully_generated.json")

    if not data_file.exists():
        print(f"❌ OH NO! I can't find the data file at: {data_file}")
        print("Make sure you've run the final 'generate_term_pairs.py' script first!")
        return

    print(f"✅ Found the data at {data_file}! Let the 6-part analysis... BEGIN!")
    with open(data_file, 'r') as f:
        all_pairs = json.load(f)

    # --- Sort pairs by category AND then by type! A double-decker sorter! ---
    categorized_pairs = defaultdict(lambda: defaultdict(list))
    for pair in all_pairs:
        categorized_pairs[pair['category']][pair['type']].append(pair)

    # --- Now, let's analyze each bucket within each bucket! ---
    for category, types in categorized_pairs.items():
        print(f"\n\n--- 🔬 Analyzing Category: {category.upper()} 🔬 ---")
        
        for pair_type, pairs in types.items():
            print(f"\n  ---Analyzing Pair Type: '{pair_type}'---")

            total_pairs = len(pairs)
            case_difference_count = 0
            truly_different_pairs = []

            for pair in pairs:
                term = pair['term']
                counterpart = pair['counterpart']

                if term.lower() == counterpart.lower():
                    case_difference_count += 1
                else:
                    truly_different_pairs.append(pair)

            percentage = (case_difference_count / total_pairs) * 100 if total_pairs > 0 else 0

            print(f"  📊 {pair_type.replace('_', ' ').title()} Analysis Complete! 📊")
            print(f"  Total pairs analyzed: {total_pairs}")
            print(f"  Pairs that are just case differences: {case_difference_count}")
            print(f"  Percentage of pairs that are just case differences: {percentage:.2f}%")

            print(f"\n  --- ✨ Salient Examples for {category.title()} ({pair_type.replace('_', ' ').title()}) ✨ ---")
            if truly_different_pairs:
                num_examples = min(5, len(truly_different_pairs))
                random_examples = random.sample(truly_different_pairs, num_examples)
                for i, pair in enumerate(random_examples):
                    print(f"    Example {i+1}:")
                    print(f"      Term       : {pair['term']}")
                    print(f"      Counterpart: {pair['counterpart']}")
            else:
                print("    No truly different pairs were found in this sub-category!")

    print("\n\n🎉 AHAHAHA! The ultra-specific, 6-part analysis is complete! The data is so beautifully organized!")

if __name__ == "__main__":
    analyze_term_pairs_by_category_and_type()