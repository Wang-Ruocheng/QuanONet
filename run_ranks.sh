#!/bin/bash

seed_list=(0 2 3 4)
max_jobs=5
job_count=0

for seed in "${seed_list[@]}"; do
  # 创建以seed命名的文件夹
  mkdir -p "seed${seed}"
  for rank in {1..16}; do
    log_file="seed${seed}/output_rank${rank}.log"
    echo "开始运行:seed=${seed}, ham_rank=${rank}"
    nohup python -u train.py \
      --operator Homogeneous \
      --num_qubits 5 \
      --if_trainable_freq true \
      --model_type QuanONet \
      --random_seed ${seed} \
      --prefix hamrank_quanonet \
      --num_epochs 1000 \
      --ham_rank ${rank} \
      > "${log_file}" 2>&1 &
    ((job_count++))
    if ((job_count >= max_jobs)); then
      wait
      job_count=0
    fi
    echo "日志实时输出：${log_file}（可用 tail -f 查看）"
  done
done
wait
echo "所有任务已完成。"