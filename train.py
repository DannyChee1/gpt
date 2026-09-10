import torch

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

data = DataLoader(B=train_config.batch_size, T=train_config.seq_len, data_path=train_config.data_path)

model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
for iter in range(train_config.max_steps):
    x, y = data.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {iter} loss: {loss.item()}")
