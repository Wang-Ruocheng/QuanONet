#!/bin/bash

# ==============================================================================
# Script Name: reproduce_sec54.sh
# Description: Reproduce Section 5.4 (Hamiltonian Ablation Studies).
#              Includes: Pauli Basis (Fig 11), Spectral Radius (Fig 11), 
#                        and Spectral Degeneracy (Fig 10).
# Usage:       ./scripts/reproduce_sec54.sh [GPU_ID]
# ==============================================================================

GPU_ID=$1
PREFIX="Sec54_Ablation"

# Common Hyperparameters
OPERATOR="Inverse"
MODEL="QuanONet"
NUM_TRAIN=1000
TRAIN_SAMPLE=10
NUM_TEST=1000
TEST_SAMPLE=100
BATCH_SIZE=100
LR=0.0001
EPOCHS=1000
IF_TF="true"
SEEDS=(0 1 2 3 4)

# Setup Device
if [ -n "$GPU_ID" ]; then
    GPU_FLAG="--gpu ${GPU_ID}"
    echo "🔧 Specified running device: GPU ${GPU_ID}"
else
    GPU_FLAG=""
    echo "🤖 Using default device (CPU for Quantum)"
fi

echo "🚀 Starting Section 5.4 Hamiltonian Ablation Experiments..."

# ==============================================================================
# Experiment 1: Pauli Basis Ablation (Fig 11 Left)
# Qubits: 5 | Net Size: 20 2 10 2 | Sweep: ham_pauli
# ==============================================================================
N_Q_1=5
NET_SIZE_1="20 2 10 2"
PAULI_LIST=("X" "Y" "Z")

echo "----------------------------------------------------------------"
echo "▶ Branch 1: Pauli Basis Traversal (X, Y, Z)"
echo "----------------------------------------------------------------"

for PAULI in "${PAULI_LIST[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "  Running Pauli: ${PAULI} | Seed: ${SEED}"
        
        python main.py \
            --model_type "${MODEL}" --operator "${OPERATOR}" \
            --num_qubits ${N_Q_1} --net_size ${NET_SIZE_1} \
            --if_trainable_freq "${IF_TF}" \
            --ham_pauli "${PAULI}" \
            --num_train ${NUM_TRAIN} --train_sample_num ${TRAIN_SAMPLE} \
            --num_test ${NUM_TEST} --test_sample_num ${TEST_SAMPLE} \
            --batch_size ${BATCH_SIZE} --learning_rate ${LR} \
            --seed "${SEED}" --prefix "${PREFIX}" ${GPU_FLAG} \
            > /dev/null 2>&1
    done
done

# ==============================================================================
# Experiment 2: Spectral Radius Ablation (Fig 11 Right)
# Qubits: 5 | Net Size: 20 2 10 2 | Sweep: ham_bound [-1, 1] to [-10, 10]
# ==============================================================================
echo "----------------------------------------------------------------"
echo "▶ Branch 2: Spectral Radius Traversal (Bound -1~1 to -10~10)"
echo "----------------------------------------------------------------"

for BOUND in {1..10}; do
    for SEED in "${SEEDS[@]}"; do
        echo "  Running Bound: [-${BOUND}, ${BOUND}] | Seed: ${SEED}"
        
        python main.py \
            --model_type "${MODEL}" --operator "${OPERATOR}" \
            --num_qubits ${N_Q_1} --net_size ${NET_SIZE_1} \
            --if_trainable_freq "${IF_TF}" \
            --ham_bound -${BOUND} ${BOUND} \
            --num_train ${NUM_TRAIN} --train_sample_num ${TRAIN_SAMPLE} \
            --num_test ${NUM_TEST} --test_sample_num ${TEST_SAMPLE} \
            --batch_size ${BATCH_SIZE} --learning_rate ${LR} \
            --seed "${SEED}" --prefix "${PREFIX}" ${GPU_FLAG} \
            > /dev/null 2>&1
    done
done

# ==============================================================================
# Experiment 3: Spectral Degeneracy Ablation (Fig 10)
# Qubits: 2 | Net Size: 50 2 50 2 | Sweep: ham_diag
# ==============================================================================
N_Q_3=2
NET_SIZE_3="50 2 50 2"

# Note: Using array of strings to handle multi-parameter arguments correctly
DIAG_LIST=(
    "-5 5 5 5"
    "-5 -5 -5 5"
    "-5 0 0 5"
    "-5 -2.5 2.5 5"
)

echo "----------------------------------------------------------------"
echo "▶ Branch 3: Spectral Degeneracy Traversal (ham_diag)"
echo "----------------------------------------------------------------"

# Index used for naming log files cleanly
IDX=1
for DIAG in "${DIAG_LIST[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "  Running Diag [${DIAG}] | Seed: ${SEED}"
        
        # Disable parameter splitting issues by passing $DIAG unquoted to argparse
        python main.py \
            --model_type "${MODEL}" --operator "${OPERATOR}" \
            --num_qubits ${N_Q_3} --net_size ${NET_SIZE_3} \
            --if_trainable_freq "${IF_TF}" \
            --ham_diag ${DIAG} \
            --num_train ${NUM_TRAIN} --train_sample_num ${TRAIN_SAMPLE} \
            --num_test ${NUM_TEST} --test_sample_num ${TEST_SAMPLE} \
            --batch_size ${BATCH_SIZE} --learning_rate ${LR} \
            --seed "${SEED}" --prefix "${PREFIX}" ${GPU_FLAG} \
            > /dev/null 2>&1
    done
    ((IDX++))
done

echo "✅ Section 5.4 Ablation experiments completed."