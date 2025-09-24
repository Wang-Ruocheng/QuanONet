import os
os.environ["DDE_BACKEND"] = "pytorch"
device_id = 1  # 指定GPU编号
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
from __init__dde import *
output_file = f"dairy"
checkpoint_file = "checkpoints_dde"
operators = ["Inverse"]
# operators = ["Inverse", "Homogeneous", "Nonlinear"]
# operators = ["Advection"]
# operators = ["RDiffusion"]
# operators = ["RDiffusion", "Advection"]
# operators = ["Darcy"]
# model_types = ["FNN", "DeepONet"]
model_types = ["DeepONet"]
# model_types = ["FNN"]
batch_size = 100
learning_rate = 0.0001
if_adalr = False
if_batch = True
if_shuffle = True
if_keep = False
# if_save = False
if_save = True
# if_aligned = True
display_every = 10
# seeds = [0]
seeds = [0, 1, 2, 3, 4]
original_stdout = sys.stdout
original_stderr = sys.stderr
verbose = 1
depths = [2, 3, 4, 5, 6]
# depths = [6]
widths = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 只暴露一个GPU，始终用cuda:0
    print(f"Using GPU: {device_id}")
DE_dict = {"Inverse": "ODE", "Nonlinear": "ODE", "Homogeneous":"ODE","RDiffusion": "PDE", "Advection": "PDE", "Darcy": "PDE"}
num_epoch_dict = {"ODE": 1000, "PDE": 100}
for depth in depths:
    for width in widths:
        for seed_num in seeds:
            for model_type in model_types:
                for operator in operators:
                    net_size = (depth, width, depth, width)
                    output_path = f"melt_deeponet_dim4/melt_{model_type}_{net_size}_{seed_num}.log"
                    if not os.path.exists(output_path) and if_save:
                        print(f"Running {operator} with depth {depth}, width {width}, seed {seed_num}, model {model_type}")
                        num_epoch = num_epoch_dict[DE_dict[operator]]
                        if DE_dict[operator] == "PDE":
                            num_train=1000
                            num_test=1000
                            train_sample_num=100
                            test_sample_num=500 if operator == 'Darcy' else 1000
                            net_size_dict = {"DeepONet":(3, 15, 3, 15), "FNN":(3, 16)}
                        else:
                            num_train=1000
                            train_sample_num=10
                            num_test=1000
                            test_sample_num=100
                            net_size_dict = {"DeepONet":(depth, width, depth, width), "FNN":(2, 10)}
                        num_iter = num_epoch*(num_train*train_sample_num//batch_size)
                        num_points = 25 if operator == 'Darcy' else 100
                        net_size = net_size_dict[model_type]
                        np.random.seed(seed_num)
                        torch.manual_seed(seed_num)
                        print(f"Seed: {seed_num}, operator: {operator}, Model: {model_type}")
                        if DE_dict[operator] == "PDE":
                            generate_data = lambda x, y: generate_PDE_Operator_data(x, y, operator=operator)
                        else:
                            generate_data = lambda x, y: generate_ODE_Operator_data(operator, x, y, num_points)
                        encode = PDE_encode if DE_dict[operator] == "PDE" else ODE_encode
                        train_branch_input, train_trunk_input, train_output, test_branch_input, test_trunk_input, test_output = encode(generate_data, num_train, num_test, num_points, train_sample_num, test_sample_num)
                        X_train = (train_branch_input.astype(np.float32), train_trunk_input.astype(np.float32))
                        X_test = (test_branch_input.astype(np.float32), test_trunk_input.astype(np.float32))
                        y_train = train_output.astype(np.float32)
                        y_test = test_output.astype(np.float32)
                        data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
                        # output_path = f"training_{operator}_{model_type}_1_{seed_num}.log"
                        if not os.path.exists(output_path) or not if_save:
                            dim_x = 1 if DE_dict[operator] == "ODE" else 2
                            if model_type == "DeepONet":
                                branch_size = [train_branch_input.shape[1]]
                                for _ in range(net_size[0]):
                                    branch_size.append(net_size[1])
                                branch_size[-1] = 4
                                trunk_size = [dim_x]
                                for _ in range(net_size[2]):
                                    trunk_size.append(net_size[3])
                                trunk_size[-1] = 4
                                net = DeepONet_nobias(branch_size, trunk_size, "tanh", "Glorot normal",)
                            elif model_type == "FNN":
                                layers_size = [dim_x + train_branch_input.shape[1]]
                                for _ in range(net_size[0]+1):
                                    layers_size.append(net_size[1])
                                layers_size.append(1)
                                net = FNN_re(layers_size, nn.Tanh(), torch.nn.init.xavier_normal_)
                            print(f"Data generated, train_num: {len(train_output)}, test_num: {len(test_output)}, params num: {sum(p.numel() for p in net.parameters())}")
                            model = dde.Model(data, net)
                            model.compile("adam", lr=learning_rate)
                            if if_batch:
                                losshistory, train_state = model.train(batch_size=batch_size, display_every = display_every, iterations=num_iter, verbose = verbose)
                            else:
                                losshistory, train_state = model.train(display_every = display_every, iterations=num_iter, verbose = verbose)
                            dir_name = os.path.dirname(output_path)
                            if dir_name:
                                os.makedirs(dir_name, exist_ok=True)
                            if not if_keep and if_save:
                                with open(output_path, 'w') as file:
                                    file.truncate(0)
                            if if_save:
                                try:
                                    reset_logging()  # 重置日志记录器
                                    logging.basicConfig(filename=output_path, level=logging.INFO, format='%(message)s')
                                    logger = logging.getLogger()
                                    sys.stdout = StreamToLogger(logger, logging.INFO)
                                    sys.stderr = StreamToLogger(logger, logging.ERROR)
                                except Exception as e:
                                    print(f"Failed to set up logging: {e}")
                                print(f"Training {model_type} with size {net_size}, params num: {sum(p.numel() for p in net.parameters())},  train_num: {len(train_output)}, test_num: {len(test_output)}, if_adalr: {if_adalr}, if_batch: {if_batch}, if_shuffle: {if_shuffle}, batch_size: {batch_size}, learning_rate: {learning_rate}")
                                train_loss = [losshistory.loss_train[j][0] for j in range(num_iter//display_every)]
                                test_loss = [losshistory.loss_test[j][0] for j in range(num_iter//display_every)]
                                for i in range(num_epoch):
                                    unit = num_train*train_sample_num//(batch_size * display_every)
                                    print(f"Epoch {i}: {np.mean(train_loss[i*unit:(i+1)*unit])}, {test_loss[i*unit+unit-1]}")
                                checkpoint_file_name = f"{checkpoint_file}/{operator}/{operator}_{model_type}_{net_size}_seed{seed_num}.ckpt"
                                dir_name = os.path.dirname(checkpoint_file_name)
                                if dir_name:
                                    os.makedirs(dir_name, exist_ok=True)
                                torch.save({'model_state_dict': net.state_dict()}, checkpoint_file_name)
                            sys.stdout = original_stdout
                            sys.stderr = original_stderr
                            print(f"{output_path} finished")