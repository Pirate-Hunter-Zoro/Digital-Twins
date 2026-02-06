#!/bin/bash
#SBATCH --job-name=clean_artifacts
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH -o slurm_jobs/logs/clean_artifacts_out.txt
#SBATCH -e slurm_jobs/logs/clean_artifacts_err.txt

cd /media/studies/ehr_study/analysis/mferguson/
echo "Removing artifacts..."
rm -rf patient_json/ && rm -rf sliced_patient_json/ && rm -rf unsliced_patient_json/ && rm -rf artifacts/
echo "Artifacts removed!"