#!/bin/bash

# ==============================================================================
# Script Name: reproduce_table4.sh
# Description: Reproduce Table 4 benchmarks for QuanONet paper.
#              Iterates over Models, Operators, Frequencies, Scales, and Seeds.
# Usage:       ./scripts/reproduce_table4.sh [GPU_ID]
# Example:     ./scripts/reproduce_table4.sh 0
# ==============================================================================

# 1. Configuration
# ----------------
GPU_ID=$1
LOG_DIR="Table4_Reproduction/dairy"
mkdir -p "$LOG_DIR"

# Common Hyperparameters
NUM_TRAIN=1000
TRAIN_SAMPLE=10
NUM_TEST=1000
TEST_SAMPLE=100
BATCH_SIZE=100
LR=0.0001
PREFIX="Table4_Reproduction"
SEEDS=(0 1 2 3 4)

Iteration Lists
OPERATORS=("Inverse" "Homogeneous" "Nonlinear" "RDiffusion" "Advection" "Darcy")
MODELS=("HEAQNN" "QuanONet" "DeepONet" "FNN")
FREQUENCIES=("true" "false")
SCALES=(0.1 0.01 0.001)


# ==============================================================================
# 2. Main Execution Loop
# ==============================================================================
if [ -n "$GPU_ID" ]; then
    GPU_FLAG="--gpu ${GPU_ID}"
    echo "🔧 Specified running device: GPU ${GPU_ID}"
else
    GPU_FLAG=""
    echo "🤖 For unspecified devices: The smart default policy of main.py will be used (Quantum->CPU, Classical->GPU)"
fi

echo "🚀 Starting Table 4 Reproduction Experiment on GPU ${GPU_ID}..."

for OP in "${OPERATORS[@]}"; do
    
    # --- A. Determine Problem Type (ODE vs PDE) ---
    if [[ "$OP" == "Inverse" || "$OP" == "Homogeneous" || "$OP" == "Nonlinear" ]]; then
        PROB_TYPE="ODE"
        EPOCHS=1000
        # ODE specific resolution defaults (if needed, otherwise relying on main.py defaults)
        PTS=100
        PTS_0=100
    else
        PROB_TYPE="PDE"
        EPOCHS=100
        # PDE specific resolution defaults
        PTS=100
        PTS_0=100
    fi

    echo "----------------------------------------------------------------"
    echo "▶ Operator: ${OP} (${PROB_TYPE}) | Epochs: ${EPOCHS}"
    echo "----------------------------------------------------------------"

    for MODEL in "${MODELS[@]}"; do
        
        # === Branch 1: Quantum Models (Iterate Freq & Scale) ===
        if [[ "$MODEL" == "QuanONet" || "$MODEL" == "HEAQNN" ]]; then
            
            for IF_TF in "${FREQUENCIES[@]}"; do
                
                # --- Map Net Size based on Model, Type & TF Strategy ---
                NET_SIZE=""
                if [[ "$PROB_TYPE" == "ODE" ]]; then
                    if [[ "$MODEL" == "QuanONet" ]]; then
                        if [[ "$IF_TF" == "true" ]]; then NET_SIZE="20 2 10 2"; else NET_SIZE="20 2 20 2"; fi
                    elif [[ "$MODEL" == "HEAQNN" ]]; then
                        if [[ "$IF_TF" == "true" ]]; then NET_SIZE="32 2"; else NET_SIZE="40 2"; fi
                    fi
                else # PDE
                    if [[ "$MODEL" == "QuanONet" ]]; then
                        if [[ "$IF_TF" == "true" ]]; then NET_SIZE="40 2 20 2"; else NET_SIZE="40 2 40 2"; fi
                    elif [[ "$MODEL" == "HEAQNN" ]]; then
                        if [[ "$IF_TF" == "true" ]]; then NET_SIZE="64 2"; else NET_SIZE="80 2"; fi
                    fi
                fi
                
                for SCALE in "${SCALES[@]}"; do
                    for SEED in "${SEEDS[@]}"; do

                        echo "  [Quantum] Running ${MODEL} | TF=${IF_TF} | Size=[${NET_SIZE}] | Scale=${SCALE} | Seed=${SEED}"
                        
                        # Execute Quantum Training
                        python main.py \
                            --model_type "${MODEL}" \
                            --operator "${OP}" \
                            --net_size ${NET_SIZE} \
                            --if_trainable_freq "${IF_TF}" \
                            --scale_coeff "${SCALE}" \
                            --num_train ${NUM_TRAIN} --train_sample_num ${TRAIN_SAMPLE} \
                            --num_test ${NUM_TEST} --test_sample_num ${TEST_SAMPLE} \
                            --batch_size ${BATCH_SIZE} \
                            --num_epochs ${EPOCHS} \
                            --learning_rate ${LR} \
                            --num_points ${PTS} --num_points_0 ${PTS_0} \
                            --seed "${SEED}" \
                            --prefix "${PREFIX}" \
                            ${GPU_FLAG} \
                            > /dev/null 2>&1
                    done
                done
            done

        # === Branch 2: Classical Models (No Freq/Scale Loop) ===
        else 
            # --- Map Net Size for Classical Models ---
            NET_SIZE=""
            if [[ "$PROB_TYPE" == "ODE" ]]; then
                if [[ "$MODEL" == "DeepONet" ]]; then NET_SIZE="2 10 2 10"; fi
                if [[ "$MODEL" == "FNN" ]];      then NET_SIZE="2 10"; fi
            else # PDE
                if [[ "$MODEL" == "DeepONet" ]]; then NET_SIZE="3 15 3 15"; fi
                if [[ "$MODEL" == "FNN" ]];      then NET_SIZE="3 16"; fi
            fi

            for SEED in "${SEEDS[@]}"; do
                echo "  [Classical] Running ${MODEL} | Size=[${NET_SIZE}] | Seed=${SEED}"
                
                # Execute Classical Training
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
                    > /dev/null 2>&1
            done
        fi
    done
done

echo "✅ All experiments completed. Logs saved in ${LOG_DIR}"