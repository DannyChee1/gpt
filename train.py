import torch
import tiktoken

from model import GPT, GPTConfig

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

num_return_sequences = 10
max_length = 100

enc = tiktoken.get_encoding("gpt2")

with open("text.txt", "r") as f:
    text = f.read()
text = text[:10000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1], device=device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

model = GPT(GPTConfig())
model.to(device)
logits = model(x)
print(logits.shape)
