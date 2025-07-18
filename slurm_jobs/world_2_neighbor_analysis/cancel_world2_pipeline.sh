#!/bin/bash
echo '--- 🛑 Sending stop signal to all World 2 pipeline jobs! 🛑 ---'
scancel 1323906 # W2_Pipeline_gru_V-6_S-50
scancel 1323907 # W2_Pipeline_transformer_V-6_S-50
