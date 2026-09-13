import torch
import time

from config import GPTConfig, TrainConfig
from data import DataLoader
from model import GPT


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = get_device()
print(f"Using device: {device}")

train_config = TrainConfig()

torch.manual_seed(train_config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(train_config.seed)

data = DataLoader(B=train_config.batch_size, T=train_config.seq_len, data_path=train_config.data_path)

torch.set_float32_matmul_precision("high") # use TF32 instead of FP32 for faster computation

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
for iter in range(train_config.max_steps):
    t0 = time.time()
    x, y = data.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_per_sec = train_config.batch_size * train_config.seq_len / dt
    print(f"step {iter} loss: {loss.item()} time: {dt*1000:.2f}ms tokens/sec: {tokens_per_sec:.2f}")
