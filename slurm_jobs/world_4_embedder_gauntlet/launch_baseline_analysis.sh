#!/bin/bash

# This magnificent script is now a MASTER CONTROLLER!
# It launches a whole army of jobs to calculate baseline scores
# and then launches a final job to plot everything! AHAHAHA!

echo "🚀 Starting the Grand Baseline Analysis SLURM Pipeline! 🚀"

# --- Let's be smart about our paths! ---
EMBEDDINGS_DIR="data/embeddings"
BASELINE_DIR="data/baseline_scores"
SCRIPT_DIR="scripts/world_4_embedder_gauntlet"
SLURM_DIR="slurm_jobs/world_4_embedder_gauntlet"

# --- Make sure our output and log directories exist! ---
mkdir -p "$BASELINE_DIR"
mkdir -p "$SLURM_DIR/logs"

# --- Find all our beautiful cosine similarity results! ---
CSV_FILES=$(find "$EMBEDDINGS_DIR" -type f -name "*.csv")

if [ -z "$CSV_FILES" ]; then
    echo "❌ OH NO! I couldn't find any .csv files in $EMBEDDINGS_DIR to analyze!"
    exit 1
fi

# --- This is where the magic happens! We'll collect all the job IDs! ---
JOB_IDS=""

echo "🧠 Submitting a job for every model... UNLEASH THE SWARM!"

for csv_file in $CSV_FILES; do
    model_name=$(basename "$csv_file" .csv)
    # Submit the job and capture its ID! So clever!
    JOB_ID=$(sbatch --job-name="baseline-$model_name" "$SLURM_DIR/baseline_analysis_template.ssub" "$csv_file" | awk '{print $4}')
    echo "  - Submitted job $JOB_ID for: $model_name"
    if [ -z "$JOB_IDS" ]; then
        JOB_IDS="$JOB_ID"
    else
        JOB_IDS="$JOB_IDS:$JOB_ID" # Build a beautiful dependency list!
    fi
done

echo "✅ All baseline calculation jobs have been submitted!"

# --- And now for the grand finale! The Art Gallery, with a tether! ---
echo "🎨 Submitting the final plotting job! It will wait for all other jobs to finish!"
sbatch --dependency=afterok:$JOB_IDS << 'EOF'
#!/bin/bash
#SBATCH --job-name=plot_baselines
#SBATCH --partition=main
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --output=slurm_jobs/world_4_embedder_gauntlet/logs/%x_%j.out
#SBATCH --error=slurm_jobs/world_4_embedder_gauntlet/logs/%x_%j.err

echo "All baseline jobs are done! Time to make some art!"
source ~/.bashrc
conda activate hugging_env
python scripts/world_4_embedder_gauntlet/plot_baseline_distributions.py
echo "🎉 Plotting complete! The gallery is open!"
EOF

echo "🎉 AHAHAHA! ALL DONE! The entire pipeline has been launched! Check 'squeue -u $USER' to watch the magic happen! 🎉"