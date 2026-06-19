#!/bin/bash

eval "$(/opt/apps/easybuild/software/Anaconda3/2025.06-0/bin/conda shell.bash hook)"

set -o errexit
set -o nounset
set -o pipefail

ENV_BASE="/media/studies/ehr_study/analysis/mferguson/venvs"
export CONDA_PKGS_DIRS=$ENV_BASE/"conda_pkgs"

# Create environment for the main pipeline
MAIN_ENV=$ENV_BASE/embedder_pipeline

if [ -f "$MAIN_ENV/conda-meta/history" ]; then
    echo "Main conda environment already created"
else
    # prefix flag and auto-yes on proceeding
    conda create -p $MAIN_ENV python=3.11 -y
fi

conda activate $MAIN_ENV

export PYTHONNOUSERSITE=1
python -m pip install scikit-learn==1.7.1\
                        numpy==2.2.6\
                        torch==2.7.1\
                        vllm==0.10.0\
                        fastapi==0.116.1\
                        starlette==0.47.2\
                        uvicorn==0.35.0\
                        prometheus-fastapi-instrumentator==7.1.0\
                        transformers==4.55.4\
                        pandas==2.3.1\
                        scipy==1.16.1\
                        xgboost==3.1.1\
                        joblib==1.5.1\
                        matplotlib==3.10.5\
                        seaborn==0.13.2\
                        sentence-transformers==5.1.0\
                        pyarrow==21.0.0\
                        requests==2.32.4\
                        python-dotenv==1.1.1\
                        tqdm==4.67.3\
                        pytest\
                        ipykernel

# Smoke check: prove the heavy packages import before claiming success.
# errexit aborts the script loud if any import fails. sklearn must read 1.7.1.
echo "Verifying main pipeline environment..."
python -c "import numpy, pandas, scipy, sklearn, torch, transformers, sentence_transformers, vllm, xgboost, joblib, matplotlib, pyarrow; print(f'main env OK | scikit-learn {sklearn.__version__}')"

# Create environment for causal random forest investigation
CAUSAL_ENV=$ENV_BASE/causal_forest_env

if [ -f "$CAUSAL_ENV/conda-meta/history" ]; then
    echo "Causal conda environment already created"
else
    # prefix flag and auto-yes on proceeding
    conda create -p $CAUSAL_ENV python=3.11 -y
fi

conda activate $CAUSAL_ENV

python -m pip install econml==0.16.0\
                    scikit-learn==1.6.1\
                    shap\
                    lightgbm\
                    joblib\
                    pyarrow\
                    ipykernel\
                    python-dotenv\
                    requests\
                    statsmodels\
                    numpy\
                    pandas\
                    scipy\
                    matplotlib

# Smoke check: prove the lean env imports before claiming success.
# sklearn must read 1.6.1 here (econml caps it below 1.7).
echo "Verifying causal forest environment..."
python -c "import econml, sklearn, shap, lightgbm, statsmodels, numpy, pandas, scipy, matplotlib, pyarrow; print(f'causal env OK | scikit-learn {sklearn.__version__}')"