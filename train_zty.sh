#!/bin/bash
 
# --- Script Config ---
 
set -e
set -o pipefail

# Maximum concurrent jobs limit
MAX_CONCURRENT_JOBS=2

# Relative path of the training logs, uncomment to set it.
LOG_DIR="sep21"
 
PREFIX="melt_quanonet_dim4"

# Work directory
WORK_PATH="/mnt/nas-new/home/yange/wangruocheng/QON_wrc"
 
# Conda env name or directory
CONDA_PATH="/mnt/nas-new/home/yange/wangruocheng/mindquan"
 
# --- Params Config ---
 
# Oprator to be trained
OPERATOR="Inverse"

# NN Model to be trained
MODELTYPE="QuanONet"

# Scale coefficent
SCALECOEFF=0.001

# is_tf
ISTF="true"

# num of qubits
NUMQUBITS=2

# Define all net_size configs to be tested in an array.
NET_SIZES=(
	"200 2 10 2"
	"200 2 20 2"
	"200 2 30 2"
	"200 2 40 2"
	"200 2 50 2"
	"200 2 60 2"
)

# Define the range of random seeds
SEEDS=($(seq 0 4))

# --- Main Logic ---

echo "Initializing Conda..."
eval "$(conda shell.bash hook)"

echo "Activating Conda environment: $CONDA_PATH"
conda activate "$CONDA_PATH" || {
    echo "Error: Failed to activate Conda environment: $CONDA_PATH"
    echo "Please check if the environment exists and path is correct."
    exit 1
}

echo "Current Conda environment: $(conda env list | grep '*')"

cd $WORK_PATH || {
    echo "Error: Failed to change to work directory: $WORK_PATH"
    exit 1
}

echo "Starting training script..."
echo "--------------------------------"
echo "Model Type: $MODELTYPE"
echo "Number of Qubits: $NUMQUBITS"
echo "Net Sizes to test: ${#NET_SIZES[@]}"
echo "Seeds per run: ${SEEDS[@]}"
echo "Prefix for logs: $PREFIX"
echo "Operator: $OPERATOR"
echo "Scale Coefficient: $SCALECOEFF"
echo "Is Trainable Frequency: $ISTF"
echo "--------------------------------"

mkdir -p "$LOG_DIR"

for net_size in "${NET_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
                if (( $(jobs -p | wc -l) >= MAX_CONCURRENT_JOBS )); then
                        wait -n
                fi

                sanitized_net_size=$(echo "$net_size" | tr ' ' '_')

                LOGFILE="${LOG_DIR}/train_${MODELTYPE}_netsize_${sanitized_net_size}_seed_${seed}.log"

                echo "Starting run: netsize=($net_size), seed=$seed, Logging to $LOGFILE"

                nohup python -u train_ODE.py \
                        --operator "$OPERATOR" \
                        --model_type "$MODELTYPE" \
                        --scale_coeff "$SCALECOEFF" \
                        --if_trainable_freq "$ISTF" \
                        --num_qubits $NUMQUBITS \
                        --net_size $net_size \
                        --random_seed "$seed" \
                        --prefix "$PREFIX" > "$LOGFILE" 2>&1 &
        done
done

echo "All training jobs have been launched. Waiting for completion..."
wait
echo "All training jobs have finished."

