#!/bin/bash

# ==============================================================================
# Script Name: reproduce_table7.sh
# Description: Reproduce Table 7 benchmarks (Parameter Efficiency Analysis).
#              Grid search for TF-QuanONet (Hidden dims) & DeepONet (Width/Depth).
# Operator:    Inverse (ODE)
# Usage:       ./scripts/reproduce_table7.sh [GPU_ID]
# ==============================================================================

# 1. Configuration
# ----------------
GPU_ID=$1
PREFIX="Table7_Reproduction"
LOG_DIR="${PREFIX}/dairy"
mkdir -p "$LOG_DIR"

# Fixed Hyperparameters
OPERATOR="Inverse"
NUM_TRAIN=1000
TRAIN_SAMPLE=10
NUM_TEST=1000
TEST_SAMPLE=100
BATCH_SIZE=100
LR=0.0001
EPOCHS=1000
SEEDS=(0 1 2 3 4)

# Iteration Lists (Parameter Grid)
# TF-QuanONet
HB_LIST=(50 100 150 200)
HT_LIST=(10 20 30 40 50 60 100 150 200 300)

# DeepONet
DEPTH_LIST=(2 3 4 5)
WIDTH_LIST=(4 8 16 32 64 128 256 512 1024)

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

echo "🚀 Starting Table 7 Reproduction (Architecture Search)..."
echo "📂 Output Directory: ${PREFIX}"

# ==============================================================================
# 3. Experiment Loop 1: TF-QuanONet
# ==============================================================================
MODEL="QuanONet"
# TF-QuanONet settings
NUM_QUBITS=2
IF_TF="true"

echo "----------------------------------------------------------------"
echo "▶ Branch 1: ${MODEL} (TF=${IF_TF}, Qubits=${NUM_QUBITS})"
echo "----------------------------------------------------------------"

for HB in "${HB_LIST[@]}"; do
    for HT in "${HT_LIST[@]}"; do
        
        # Net Size format: hb 2 ht 2
        NET_SIZE="${HB} 2 ${HT} 2"
        
        for SEED in "${SEEDS[@]}"; do
            LOG_FILE="${LOG_DIR}/${OPERATOR}_${MODEL}_HB${HB}_HT${HT}_Seed${SEED}.log"
            echo "  [Quantum] Running ${MODEL} | Size=[${NET_SIZE}] | Seed=${SEED}"
            
            python main.py \
                --model_type "${MODEL}" \
                --operator "${OPERATOR}" \
                --net_size ${NET_SIZE} \
                --num_qubits ${NUM_QUBITS} \
                --if_trainable_freq "${IF_TF}" \
                --num_train ${NUM_TRAIN} --train_sample_num ${TRAIN_SAMPLE} \
                --num_test ${NUM_TEST} --test_sample_num ${TEST_SAMPLE} \
                --batch_size ${BATCH_SIZE} \
                --num_epochs ${EPOCHS} \
                --learning_rate ${LR} \
                --seed "${SEED}" \
                --prefix "${PREFIX}" \
                ${GPU_FLAG} \
                > "${LOG_FILE}" 2>&1
        done
    done
done

# ==============================================================================
# 4. Experiment Loop 2: DeepONet
# ==============================================================================
MODEL="DeepONet"
OUTPUT_DIM=4  # Explicit 5th parameter for net_size

echo "----------------------------------------------------------------"
echo "▶ Branch 2: ${MODEL} (Grid Search: Depth x Width)"
echo "----------------------------------------------------------------"

for DEPTH in "${DEPTH_LIST[@]}"; do
    for WIDTH in "${WIDTH_LIST[@]}"; do
        
        # Net Size format: depth width depth width 4
        # This ensures output dimension p=4 regardless of width
        NET_SIZE="${DEPTH} ${WIDTH} ${DEPTH} ${WIDTH} ${OUTPUT_DIM}"
        
        for SEED in "${SEEDS[@]}"; do
            LOG_FILE="${LOG_DIR}/${OPERATOR}_${MODEL}_D${DEPTH}_W${WIDTH}_Seed${SEED}.log"
            echo "  [Classical] Running ${MODEL} | Size=[${NET_SIZE}] | Seed=${SEED}"
            
            python main.py \
                --model_type "${MODEL}" \
                --operator "${OPERATOR}" \
                --net_size ${NET_SIZE} \
                --num_train ${NUM_TRAIN} --train_sample_num ${TRAIN_SAMPLE} \
                --num_test ${NUM_TEST} --test_sample_num ${TEST_SAMPLE} \
                --batch_size ${BATCH_SIZE} \
                --num_epochs ${EPOCHS} \
                --learning_rate ${LR} \
                --seed "${SEED}" \
                --prefix "${PREFIX}" \
                ${GPU_FLAG} \
                > "${LOG_FILE}" 2>&1
        done
    done
done

echo "✅ Table 7 experiments completed."