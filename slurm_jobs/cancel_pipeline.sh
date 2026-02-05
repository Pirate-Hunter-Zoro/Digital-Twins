#!/bin/bash
echo 'Killing prediction pipeline...'
scancel 1682914 # Embedding Job
scancel 1682915 # vLLM Server
scancel 1682916 # Prediction Pipeline
scancel 1682917 # Analysis Pipeline
scancel 1682913 # Orchestrator (Self)
