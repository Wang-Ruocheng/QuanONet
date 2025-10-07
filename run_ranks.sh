#!/bin/bash

bd_list=(50 100 150 200)
td_list=(10 20 30 40 50 60 70 80)
seed_list=(4)
max_jobs=4
job_count=0

for bd in "${bd_list[@]}"; do
  for td in "${td_list[@]}"; do
    for seed in "${seed_list[@]}"; do
      log_file="output_bd${bd}_td${td}_seed${seed}.log"
      json_file="train_Inverse_TF-QuanONet_2_[${bd},2,${td},2]_${seed}.json"
      if [ ! -f "${json_file}" ]; then
        echo "开始运行:bd=${bd}, td=${td}, seed=${seed}"
        nohup python -u train.py \
          --operator Inverse \
          --model_type QuanONet \
          --scale_coeff 0.001 \
          --if_trainable_freq true \
          --num_qubits 5 \
          --net_size ${bd} 2 ${td} 2 \
          --if_train true \
          --if_keep false \
          --if_save true \
          --random_seed ${seed} \
          --if_adjust_lr false \
          --prefix melt_quanonet_dim4 \
          --num_epochs 1000 \
          > "${log_file}" 2>&1 &
        ((job_count++))
        if ((job_count >= max_jobs)); then
          wait
          job_count=0
        fi
        echo "日志实时输出：${log_file}（可用 tail -f 查看）"
      else
        echo "已存在：${json_file}，跳过"
      fi
    done
  done
done
wait
