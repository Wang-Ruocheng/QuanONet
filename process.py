import re, json

log_path = "PI/dairy/train_Nonlinear_TF-QuanONet_200*10_5_[20, 2, 10, 2]_0.001_0.log"
json_path = "PI/logs/Nonlinear/train_Nonlinear_TF-QuanONet_200*10_5_[20, 2, 10, 2]_0.001_0.json"

pat_improve = re.compile(r"\b(\d+)\s*/\s*(\d+).*?Found better model:\s*Training loss improved from\s*([0-9.infINF]+)\s*to\s*([0-9.]+)", re.I)
pat_epoch = re.compile(r"Epoch\s+(\d+):\s*Train\s*=\s*([0-9.]+)", re.I)

extracted = {}

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m1 = pat_improve.search(line)
        if m1:
            ep = int(m1.group(1))
            new_loss = float(m1.group(4))
            extracted[ep] = min(new_loss, extracted.get(ep, new_loss))
            continue
        m2 = pat_epoch.search(line)
        if m2:
            ep = int(m2.group(1))
            loss = float(m2.group(2))
            # 以更小的 loss 为准
            extracted[ep] = min(loss, extracted.get(ep, loss))

# 构造新的 training_history（只保留日志中出现的 epoch）
new_history = [{"epoch": ep, "train_loss": round(loss, 6)} for ep, loss in sorted(extracted.items())]

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["training_history"] = new_history

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"已更新 {len(new_history)} 条记录到 training_history。")