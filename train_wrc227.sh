#!/usr/bin/env bash
set -euo pipefail

# 并发上限
MAX_JOBS=${MAX_JOBS:-4}

# 任务配置
OPERATOR="Nonlinear"
MODEL_TYPE="QuanONet"
IF_TF=true                  # 是否 TF（trainable freq）
NUM_QUBITS=5
# 传参用数组；日志名用格式化字符串
NET_SIZE=(20 2 10 2)
SCALE=0.001
# 多个随机种子
SEEDS=(0 1 2 3 4)
PREFIX="PI"
IF_SAVE=true
IF_KEEP=false
IF_TRAIN=true
TRAIN_SAMPLE_NUM=10

# 训练范围
START_N=100
END_N=100
STEP_N=10

START_S=1
END_S=10
STEP_S=1

# 目录
LOG_DIR="${PREFIX}/dairy"
JSON_DIR="${PREFIX}/logs"
mkdir -p "${LOG_DIR}"
mkdir -p "${JSON_DIR}"
mkdir -p "${JSON_DIR}/${OPERATOR}"

# 工具：限制并发
throttle() {
  while [ "$(jobs -r | wc -l)" -ge "${MAX_JOBS}" ]; do
    sleep 2
  done
}

# NET_SIZE 标签，如 [20, 2, 10, 2]（确保为逗号+空格）
_ns_joined=$(printf "%s, " "${NET_SIZE[@]}")
NET_SIZE_LABEL="[${_ns_joined%, }]"
TF_TAG=$([ "${IF_TF}" = "true" ] && echo "TF-" || echo "")

for IF_PI in true false; do
  PI_TAG=$([ "${IF_PI}" = "true" ] && echo "PI-" || echo "")
  for NUM_TRAIN in $(seq ${START_N} ${STEP_N} ${END_N}); do
    for TRAIN_SAMPLE_NUM in $(seq ${START_S} ${STEP_S} ${END_S}); do
    # 计算 epoch，至少为 1
    EPOCHS=$((10000000/(NUM_TRAIN*TRAIN_SAMPLE_NUM)))
    if [ "${EPOCHS}" -lt 1 ]; then EPOCHS=1; fi

    for SEED in "${SEEDS[@]}"; do
      LOG_FILE="${LOG_DIR}/train_${OPERATOR}_${PI_TAG}${TF_TAG}${MODEL_TYPE}_${NUM_TRAIN}*${TRAIN_SAMPLE_NUM}_${NUM_QUBITS}_${NET_SIZE_LABEL}_${SCALE}_${SEED}.log"
      JSON_FILE="${JSON_DIR}/${OPERATOR}/train_${OPERATOR}_${PI_TAG}${TF_TAG}${MODEL_TYPE}_${NUM_TRAIN}*${TRAIN_SAMPLE_NUM}_${NUM_QUBITS}_${NET_SIZE_LABEL}_${SCALE}_${SEED}.json"

      echo "Checking JSON: ${JSON_FILE}"
      if [ -f "${JSON_FILE}" ]; then
        echo "跳过已存在的文件：${JSON_FILE}"
        continue
      fi

      echo "Launching: PI=${IF_PI}, num_train=${NUM_TRAIN}, S=${TRAIN_SAMPLE_NUM}, seed=${SEED}, epochs=${EPOCHS}"
      nohup python -u train.py \
        --operator "${OPERATOR}" \
        --model_type "${MODEL_TYPE}" \
        --scale_coeff "${SCALE}" \
        --if_trainable_freq "${IF_TF}" \
        --num_qubits "${NUM_QUBITS}" \
        --net_size "${NET_SIZE[@]}" \
        --random_seed "${SEED}" \
        --num_epochs "${EPOCHS}" \
        --prefix "${PREFIX}" \
        --if_save "${IF_SAVE}" \
        --if_keep "${IF_KEEP}" \
        --if_train "${IF_TRAIN}" \
        --if_pi "${IF_PI}" \
        --num_train "${NUM_TRAIN}" \
        --train_sample_num "${TRAIN_SAMPLE_NUM}" \
        > "${LOG_FILE}" 2>&1 &

      throttle
    done
    done
  done
done

wait
echo "All jobs finished."