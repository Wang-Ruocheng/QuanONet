#!/bin/bash

# ==============================================================================
# Script Name: reproduce_table5.sh
# Description: Reproduce Table 5 benchmarks (Small Data Regime).
#              Fixed Scale(0.001) & TF(true) for QuanONet.
#              Specific Net Sizes for DeepONet & FNO.
# Usage:       ./scripts/reproduce_table5.sh [GPU_ID]
# ==============================================================================

# 1. Configuration
# ----------------
GPU_ID=$1
PREFIX="Table5_Reproduction"
LOG_DIR="${PREFIX}/dairy"
mkdir -p "$LOG_DIR"

# Hyperparameters (Table 5 Specific: Small Data)
NUM_TRAIN=100        # Reduced training functions
TRAIN_SAMPLE=100     # Increased samples per function
NUM_TEST=1000
TEST_SAMPLE=100
BATCH_SIZE=100
LR=0.0001

# Iteration Lists
SEEDS=(0 1 2 3 4)
OPERATORS=("Inverse" "Homogeneous" "Nonlinear")
MODELS=("QuanONet" "DeepONet" "FNO") # QuanONet here implies TF-QuanONet

# ==============================================================================
# 2. Setup Device Flag
# ==============================================================================
if [ -n "$GPU_ID" ]; then
    GPU_FLAG="--gpu ${GPU_ID}"
    echo "🔧 Specified running device: GPU ${GPU_ID}"
else
    GPU_FLAG=""
    echo "🤖 For unspecified devices: Using smart default (Quantum->CPU, Classical->GPU)"
fi

echo "🚀 Starting Table 5 Reproduction (Small Data Regime)..."
echo "📂 Output Directory: ${PREFIX}"

# ==============================================================================
# 3. Main Loop
# ==============================================================================

for OP in "${OPERATORS[@]}"; do
    
    # Table 5 only focuses on ODEs
    PROB_TYPE="ODE"
    EPOCHS=1000
    PTS=100
    PTS_0=100

    echo "----------------------------------------------------------------"
    echo "▶ Operator: ${OP} | Epochs: ${EPOCHS} | Train: ${NUM_TRAIN}x${TRAIN_SAMPLE}"
    echo "----------------------------------------------------------------"

    for MODEL in "${MODELS[@]}"; do
        
        # --- Configure Model Specific Settings ---
        NET_SIZE=""
        EXTRA_ARGS=""
        
        if [[ "$MODEL" == "QuanONet" ]]; then
            # TF-QuanONet Setup
            # Scale=0.001, Freq=True, Size=20 2 10 2
            NET_SIZE="20 2 10 2"
            SCALE="0.001"
            IF_TF="true"
            EXTRA_ARGS="--scale_coeff ${SCALE} --if_trainable_freq ${IF_TF}"
            MODEL_DESC="TF-QuanONet (Scale=${SCALE})"
            
        elif [[ "$MODEL" == "DeepONet" ]]; then
            # DeepONet Setup
            # Size=4 32 4 32
            NET_SIZE="4 32 4 32"
            MODEL_DESC="DeepONet"
            
        elif [[ "$MODEL" == "FNO" ]]; then
            # FNO Setup
            # Size=15 14 3 32 (modes, width, depth, fc_hidden)
            NET_SIZE="15 14 3 32"
            MODEL_DESC="FNO"
        fi

        for SEED in "${SEEDS[@]}"; do
            
            LOG_FILE="${LOG_DIR}/${OP}_${MODEL}_Seed${SEED}.log"
            echo "  Running ${MODEL_DESC} | Size=[${NET_SIZE}] | Seed=${SEED}"
            
            # Execute Training
            # Note: We pass ${EXTRA_ARGS} which is empty for Classical models
            python main.py \
                --model_type "${MODEL}" \
                --operator "${OP}" \
                --net_size ${NET_SIZE} \
                --num_train ${NUM_TRAIN} --train_sample_num ${TRAIN_SAMPLE} \
                --num_test ${NUM_TEST} --test_sample_num ${TEST_SAMPLE} \
                --batch_size ${BATCH_SIZE} \
                --num_epochs ${EPOCHS} \
                --learning_rate ${LR} \
                --num_points ${PTS} --num_points_0 ${PTS_0} \
                --seed "${SEED}" \
                --prefix "${PREFIX}" \
                ${GPU_FLAG} \
                ${EXTRA_ARGS} \
                > "${LOG_FILE}" 2>&1
        done
    done
done

echo "✅ Table 5 experiments completed."