#!/bin/bash

# ==============================================================================
# Script Name: reproduce_table8.sh
# Description: Reproduce Table 8 benchmarks (Parameter Efficiency / Scalability).
#              Dynamic Grid Search for TF-QuanONet based on Qubit counts.
# Operator:    Inverse (ODE)
# Usage:       ./scripts/reproduce_table8.sh [GPU_ID]
# ==============================================================================

# 1. Configuration
# ----------------
GPU_ID=$1
PREFIX="Table8_Reproduction"
LOG_DIR="${PREFIX}/dairy"
mkdir -p "$LOG_DIR"

# Common Hyperparameters
OPERATOR="Inverse"
MODEL="QuanONet"      # Specifically TF-QuanONet
IF_TF="true"          # Trainable Frequency = True
NUM_TRAIN=1000
TRAIN_SAMPLE=10
NUM_TEST=1000
TEST_SAMPLE=100
BATCH_SIZE=100
LR=0.0001
EPOCHS=1000 # ODE default
SEEDS=(0 1 2 3 4)

# Common HT List (Same for all qubit cases)
HT_LIST=(10 20 30 40)

# Qubit Cases to iterate
QUBIT_CASES=(2 5 10)

# ==============================================================================
# 2. Setup Device Flag
# ==============================================================================
if [ -n "$GPU_ID" ]; then
    GPU_FLAG="--gpu ${GPU_ID}"
    echo "🔧 Specified running device: GPU ${GPU_ID}"
else
    GPU_FLAG=""
    echo "🤖 For unspecified devices: Using smart default (Quantum->CPU)"
fi

echo "🚀 Starting Table 8 Reproduction (Qubit Scalability Analysis)..."
echo "📂 Output Directory: ${PREFIX}"

# ==============================================================================
# 3. Main Experiment Loop
# ==============================================================================

for N_Q in "${QUBIT_CASES[@]}"; do
    
    # --- Dynamic HB List Selection based on Qubits ---
    HB_LIST=()
    if [ "$N_Q" -eq 2 ]; then
        HB_LIST=(50 100)
        echo "🔵 Configuration: Qubits=2 -> HB={50, 100}"
        
    elif [ "$N_Q" -eq 5 ]; then
        HB_LIST=(20 40)
        echo "🔵 Configuration: Qubits=5 -> HB={20, 40}"
        
    elif [ "$N_Q" -eq 10 ]; then
        HB_LIST=(10 20)
        echo "🔵 Configuration: Qubits=10 -> HB={10, 20}"
    fi

    echo "----------------------------------------------------------------"
    echo "▶ Running TF-QuanONet | Qubits: ${N_Q} | HT List: ${HT_LIST[*]}"
    echo "----------------------------------------------------------------"

    for HB in "${HB_LIST[@]}"; do
        for HT in "${HT_LIST[@]}"; do
            
            # Net Size format: hb 2 ht 2
            NET_SIZE="${HB} 2 ${HT} 2"
            
            for SEED in "${SEEDS[@]}"; do
                
                LOG_FILE="${LOG_DIR}/${OPERATOR}_${MODEL}_Q${N_Q}_HB${HB}_HT${HT}_Seed${SEED}.log"
                echo "  Running Q${N_Q} | Size=[${NET_SIZE}] | Seed=${SEED}"
                
                # Execute Training
                python main.py \
                    --model_type "${MODEL}" \
                    --operator "${OPERATOR}" \
                    --net_size ${NET_SIZE} \
                    --num_qubits ${N_Q} \
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
done

echo "✅ Table 8 Qubit experiments completed."