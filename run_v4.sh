#!/bin/zsh
# run_v4.sh — Run v4 training, keep alive through lid close
cd /Users/mahizhan/Documents/Sem7/Github/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation

conda run -n battery jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=86400 \
  --ExecutePreprocessor.kernel_name=battery \
  thermal_agent_v4.ipynb

echo "TRAINING COMPLETE: $(date)"
