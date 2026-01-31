import os
import json  # [新增] 导入json模块
os.environ["DDE_BACKEND"] = "pytorch"
from __init__dde import *
output_file = f"JMLR/fno"
checkpoint_file = "JMLR/fno/checkpoints"
operators = ["Inverse", "Homogeneous", "Nonlinear"]
# operators = ["Diffusion"]
# operators = ["Inverse"]
# operators = ["Nonlinear", "Homogeneous"]
# names = ["FNN", "DeepONet"]
# names = ["DeepONet"]
# names = ["FNN"]
names = ["FNO"]
batch_size = 1
learning_rate = 0.0001
if_adalr = False
if_batch = True
if_shuffle = True
if_keep = False
# if_save = False
if_save = True
display_every = 1
# seeds = [0]
seeds = [0, 1, 2, 3, 4]
original_stdout = sys.stdout
original_stderr = sys.stderr
verbose = 1
device_id = 2

if torch.cuda.is_available():
    device = torch.device(f"cuda:{device_id}")
    print(f"Using GPU: {device_id}")
DE_dict = {"Inverse": "ODE", "Nonlinear": "ODE", "Homogeneous":"ODE","Diffusion": "PDE", "Advection": "PDE"}
num_epoch_dict = {"ODE": 1000, "PDE": 100}
net_size = [15, 14, 3] 
for seed_num in seeds:
    for name in names:
        for operator in operators:
            num_epoch = num_epoch_dict[DE_dict[operator]]
            num_train=100
            num_test=1000
            train_sample_num=100
            test_sample_num=100
            num_iter = num_epoch*(num_train//batch_size)
            # num_iter = 10000
            num_sensors = 100
            np.random.seed(seed_num)
            torch.manual_seed(seed_num)
            num_points = 25 if operator == 'Darcy' else 100
            print(f"Seed: {seed_num}, operator: {operator}, Model: {name}")
            DE_dict = {"Inverse": "ODE", "Nonlinear": "ODE", "Homogeneous":"ODE","Diffusion": "PDE", "Advection": "PDE"}
            if DE_dict[operator] == "PDE":
                generate_data = lambda x, y: generate_PDE_Operator_data(x, y, operator=operator)
            else:
                generate_data = lambda x, y: generate_ODE_Operator_data(operator, x, y, num_points)
            # encode = PDE_fncode if DE_dict[operator] == "PDE" else ODE_fncode
            encode = ODE_fncode
            train_input, train_indices, train_output, test_input, test_indices, test_output = encode(generate_data, num_train, num_test, num_sensors, train_sample_num, test_sample_num)
            X_train = train_input.astype(np.float32)
            X_test = test_input.astype(np.float32)
            y_train = train_output.astype(np.float32)
            y_test = test_output.astype(np.float32)
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            data = Double(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            log_path = f"{output_file}/dairy/{operator}/train_{operator}_{name}__{num_train}*{train_sample_num}_{net_size}_1_{seed_num}.log"
            if not os.path.exists(log_path) or not if_save:
                dim_x = 1 if DE_dict[operator] == "ODE" else 2
                net=FNO1d(modes=net_size[0], width=net_size[1], layers=net_size[2], fc_hidden=32)
                param_num = sum(p.numel() for p in net.parameters()) # 计算参数量
                print(f"Data generated, train_num: {len(train_output)}, test_num: {len(test_output)}, params num: {param_num}")
                model = dde.Model(data, net)
                model.compile("adam", lr=learning_rate)
                if if_batch:
                    losshistory, train_state = model.train(batch_size=batch_size, display_every = display_every, iterations=num_iter, verbose = verbose)
                else:
                    losshistory, train_state = model.train(display_every = display_every, iterations=num_iter, verbose = verbose)
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                if not if_keep and if_save:
                    with open(log_path, 'w') as file:
                        file.truncate(0)
                if if_save:
                    try:
                        reset_logging()  # 重置日志记录器
                        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')
                        logger = logging.getLogger()
                        sys.stdout = StreamToLogger(logger, logging.INFO)
                        sys.stderr = StreamToLogger(logger, logging.ERROR)
                    except Exception as e:
                        print(f"Failed to set up logging: {e}")
                    print(f"Training {name} with size {net_size}, params num: {param_num},  train_num: {len(train_output)}, test_num: {len(test_output)}, if_adalr: {if_adalr}, if_batch: {if_batch}, if_shuffle: {if_shuffle}, batch_size: {batch_size}, learning_rate: {learning_rate}")
                    train_loss = [losshistory.loss_train[j][0] for j in range(num_iter//display_every)]
                    test_loss = [losshistory.loss_test[j][0] for j in range(num_iter//display_every)]
                    for i in range(num_iter//(num_train//batch_size)):
                        unit = num_train//(batch_size * display_every)
                        print(f"Epoch {i}: {np.mean(train_loss[i*unit:(i+1)*unit])}, {test_loss[i*unit+unit-1]}")
                    
                    # ---------------- [新增功能] 保存 metrics 到 JSON ----------------
                    try:
                        # 1. 获取最终的 MSE (Test Loss) 和 Train Error
                        final_mse = float(losshistory.loss_test[-1][0])
                        final_train_error = float(losshistory.loss_train[-1][0])
                        
                        # 2. 准备数据字典
                        json_data = {
                            "MSE": final_mse,
                            "Train_Error": final_train_error,
                            "param_num": int(param_num)
                        }
                        
                        # 3. 确定保存目录
                        json_logs_dir = f"{output_file}/logs/{operator}"
                        os.makedirs(json_logs_dir, exist_ok=True)
                        
                        # 4. 确定文件名 (修改点：只保留文件名，不要加路径前缀！)
                        json_filename = f"train_{operator}_{name}__{num_train}*{train_sample_num}_{net_size}_1_{seed_num}.json"
                        
                        # 5. 拼接完整路径
                        json_path = os.path.join(json_logs_dir, json_filename)
                        
                        # 6. 写入文件
                        with open(json_path, 'w') as f:
                            json.dump(json_data, f, indent=4)
                        
                        # 注意：这行print会被重定向到log文件中，如果你想在控制台看到，需要临时切回stdout或者在log里确认
                        print(f"Metrics saved to JSON: {json_path}")
                        
                    except Exception as e:
                        # 建议使用sys.__stdout__强制打印到屏幕，防止被重定向掩盖
                        sys.__stdout__.write(f"Error saving JSON logs: {e}\n")
                        print(f"Error saving JSON logs: {e}")
                    # ------------------------------------------------------------------

                    # checkpoint_file_name = f"{checkpoint_file}/{operator}/{operator}_{name}_{net_size}_seed{seed_num}_bz{batch_size}.ckpt"
                    checkpoint_file_name = f"{output_file}/checkpoints/{operator}/{name}_{num_train}*{train_sample_num}_{net_size}_1_{seed_num}/final_{name}_{num_train}*{train_sample_num}_{net_size}_1_{seed_num}.ckpt"
                    dir_name = os.path.dirname(checkpoint_file_name)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)
                    torch.save({'model_state_dict': net.state_dict()}, checkpoint_file_name)
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                print(f"{log_path} finished")