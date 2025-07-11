#!/bin/bash

# This magnificent script will run our entire baseline analysis pipeline!
# First, it calculates the baseline scores for every model, then it plots them all!
# AHAHAHA! FOR SCIENCE!

echo "🚀 Starting the Grand Baseline Analysis Pipeline! 🚀"

# --- Let's be smart about our paths! ---
# We assume this script is run from the root of the Digital-Twins project directory
EMBEDDINGS_DIR="data/embeddings"
BASELINE_DIR="data/baseline_scores"
SCRIPT_DIR="scripts/world_4_embedder_gauntlet"

# --- Make sure our output directory for the new scores exists! ---
echo "🛠️ Creating the baseline scores directory (if it's not already there)!"
mkdir -p "$BASELINE_DIR"

# --- Find all our beautiful cosine similarity results! ---
# We're looking for any .csv file that doesn't have '_baseline' in its name!
# This way, we don't accidentally run the analysis on our own results! So smart!
CSV_FILES=$(find "$EMBEDDINGS_DIR" -type f -name "*.csv")

if [ -z "$CSV_FILES" ]; then
    echo "❌ OH NO! I couldn't find any .csv files in $EMBEDDINGS_DIR to analyze!"
    exit 1
fi

# --- Loop through every single file and run the baseline calculator! ---
echo "🧠 Calculating baseline scores for every model... This is the heavy lifting part!"

for csv_file in $CSV_FILES; do
    echo "  - Analyzing file: $(basename "$csv_file")"
    python "$SCRIPT_DIR/compute_baseline_term_matching.py" --input_csv "$csv_file"
done

echo "✅ All baseline scores have been calculated! Magnificent!"

# --- And now for the grand finale! The Art Gallery! ---
echo "🎨 Now, let's create the beautiful new art gallery for our baseline scores!"
python "$SCRIPT_DIR/plot_baseline_distributions.py"

echo "🎉 AHAHAHA! ALL DONE! The entire baseline analysis is complete! Check out the magnificent plots in $BASELINE_DIR! 🎉"