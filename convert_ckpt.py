import mindspore as ms
import numpy as np

# 1. 指定您的 ckpt 文件路径
ckpt_path = "quanonet_ode_inverse_result.ckpt"

# 2. 使用 MindSpore 加载权重字典
print(f"正在加载 {ckpt_path} ...")
param_dict = ms.load_checkpoint(ckpt_path)

# 3. 提取所有权重并转换为 numpy 数组
np_dict = {}
for key, param in param_dict.items():
    # 将 MindSpore 的 Tensor 转换为 numpy 数组
    np_dict[key] = param.asnumpy()
    print(f"提取参数: {key}, 形状: {np_dict[key].shape}")

# 4. 保存为 .npz 文件
save_path = "quanonet_ode_inverse_result.npz"
np.savez(save_path, **np_dict)
print(f"\n✅ 转换成功！已保存为: {save_path}")