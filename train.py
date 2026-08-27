import torch

from config import batch_size, block_size, device, max_iters, eval_interval, eval_iters, learning_rate
from dataset import get_batch, decode
from model import BigramLanguageModel

torch.manual_seed(0)

xb, yb = get_batch('train')
# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target is {target}")

torch.manual_seed(0)
m = BigramLanguageModel()
m = m.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens=400)[0].tolist()))
