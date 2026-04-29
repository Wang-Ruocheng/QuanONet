#!/bin/bash

# ==============================================================================
# Script Name: reproduce_fig9_scaling.sh
# Description: Reproduce Fig 9 / Table 13 & 14 (High-Dimensional Scaling Limit).
#              Sweeps latent dimension p from 4 to 256 for both models.
# Operator:    Antideriv (ODE)
# Usage:       ./scripts/reproduce_fig9_scaling.sh [GPU_ID]
# ==============================================================================

# 1. Configuration
# ----------------
GPU_ID=$1
PREFIX="Fig9_Scaling_Reproduction"

# Fixed Hyperparameters
OPERATOR="Antideriv"
NUM_TRAIN=1000
TRAIN_SAMPLE=10
NUM_TEST=1000
TEST_SAMPLE=100
BATCH_SIZE=100
LR=0.0001
EPOCHS=1000
SEEDS=(0 1 2 3 4)

# Latent Dimensions to sweep (p)
DIM_P_LIST=(4 8 16 32 64 128 256)

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

echo "🚀 Starting Figure 9 Reproduction (High-Dimensional Scaling)..."
echo "📂 Output Directory: ${PREFIX}"

# ==============================================================================
# 3. Experiment Loop 1: TF-QuanONet (Table 13)
# ==============================================================================
MODEL="QuanONet"
IF_TF="true"

echo "----------------------------------------------------------------"
echo "▶ Branch 1: ${MODEL} (TF=${IF_TF}) scaling from p=4 to 256"
echo "----------------------------------------------------------------"

for P in "${DIM_P_LIST[@]}"; do
    # Map dimension p to Qubit count
    case $P in
        4)   NUM_QUBITS=2 ;;
        8)   NUM_QUBITS=3 ;;
        16)  NUM_QUBITS=4 ;;
        32)  NUM_QUBITS=5 ;;
        64)  NUM_QUBITS=6 ;;
        128) NUM_QUBITS=7 ;;
        256) NUM_QUBITS=8 ;;
    esac

    # Determine Grid Search boundaries based on Table 13
    if [ "$P" -eq 4 ]; then
        HB_LIST=(50 100 150 200)
        HT_LIST=(10 20 30 40 50 60 100 150 200 300)
    elif [ "$P" -eq 8 ]; then
        HB_LIST=(100 200)
        HT_LIST=(20 40 50 100 150 200 300)
    elif [ "$P" -eq 16 ]; then
        HB_LIST=(100 200)
        HT_LIST=(50 100)
    else
        # For p >= 32, the table shows a sparse/targeted search
        HB_LIST=(100)
        HT_LIST=(50 100)
    fi

    for HB in "${HB_LIST[@]}"; do
        for HT in "${HT_LIST[@]}"; do
            # Net Size format: hb 2 ht 2 (ansatz depth is fixed at 2)
            NET_SIZE="${HB} 2 ${HT} 2"
            
            for SEED in "${SEEDS[@]}"; do
                echo "  [Quantum] p=${P} (Qubits=${NUM_QUBITS}) | Size=[${NET_SIZE}] | Seed=${SEED}"
                
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
                    > /dev/null 2>&1 || exit 1
            done
        done
    done
done

# ==============================================================================
# 4. Experiment Loop 2: DeepONet (Table 14)
# ==============================================================================
MODEL="DeepONet"
DEPTH_LIST=(2 3 4 5)

echo "----------------------------------------------------------------"
echo "▶ Branch 2: ${MODEL} scaling from p=4 to 256"
echo "----------------------------------------------------------------"

for P in "${DIM_P_LIST[@]}"; do
    OUTPUT_DIM=$P
    
    # DeepONet Width logic from Table 14: starts at p, doubles up to 1024
    WIDTH_LIST=()
    W=$P
    while [ "$W" -le 1024 ]; do
        WIDTH_LIST+=($W)
        W=$((W * 2))
    done

    for DEPTH in "${DEPTH_LIST[@]}"; do
        for WIDTH in "${WIDTH_LIST[@]}"; do
            
            # Net Size format: b_depth b_width t_depth t_width output_dim
            NET_SIZE="${DEPTH} ${WIDTH} ${DEPTH} ${WIDTH} ${OUTPUT_DIM}"
            
            for SEED in "${SEEDS[@]}"; do
                echo "  [Classical] p=${P} | Size=[${NET_SIZE}] | Seed=${SEED}"
                
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
                    > /dev/null 2>&1 || exit 1
            done
        done
    done
done

echo "✅ Figure 9 (High-Dimensional Scaling) experiments completed."