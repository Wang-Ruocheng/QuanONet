#!/bin/bash
# filepath: run_train_ODE.sh

operators=(Nonlinear)
seeds=(0 1 2 3 4)
max_jobs=1
job_count=0
scale_coeffs=(0.001 0.01 0.1)
model_type="QuanONet"

for operator in "${operators[@]}"; do
  for seed in "${seeds[@]}"; do
    for scale_coeff in "${scale_coeffs[@]}"; do
      log_file="dairy/train_${operator}_${model_type}_5_[20, 2, 20, 2]_${scale_coeff}_${seed}.log"
      json_file="logs/${operator}/train_${operator}_${model_type}_5_[20, 2, 20, 2]_${scale_coeff}_${seed}.json"
      if [ -f "${json_file}" ]; then
        echo "跳过已存在的文件：${json_file}"
        continue
      fi
        echo "开始运行：${json_file}"
        nohup python -u train_ODE.py \
          --operator "${operator}" \
          --model_type "${model_type}" \
          --scale_coeff "${scale_coeff}" \
          --if_trainable_freq false \
          --num_qubits 5 \
          --net_size 20 2 20 2 \
          --random_seed "${seed}" \
          --num_epochs 1000 \
          --if_save true \
          --if_keep false \
          --if_train true \
          > "${log_file}" 2>&1 &
        ((job_count++))
        if ((job_count >= max_jobs)); then
          wait
          echo "本轮所有任务运行结束"
          job_count=0
        fi
      done
    done
  done
wait
echo "全部任务运行结束"