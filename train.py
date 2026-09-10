import torch
import tiktoken

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("text.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos:self.current_pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_pos += B * T

        if self.current_pos + B * T + 1 > len(self.tokens):
            self.current_pos = 0
        return x, y

from model import GPT, GPTConfig

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

num_return_sequences = 10
max_length = 100

data = DataLoader(B=4, T=32)

model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for iter in range(50):
    x, y = data.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {iter} loss: {loss.item()}")

import sys; sys.exit()

