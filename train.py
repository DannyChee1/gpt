import torch

from config import BATCH_SIZE, BLOCK_SIZE, DEVICE, MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS
from dataset import get_batch, vocab_size, decode
from model import BigramLanguageModel

torch.manual_seed(0)

xb, yb = get_batch('train')
# for b in range(BATCH_SIZE):
#     for t in range(BLOCK_SIZE):
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target is {target}")

torch.manual_seed(0)
m = BigramLanguageModel(vocab_size)
m = m.to(DEVICE)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

for i in range(MAX_ITERS):
    if i % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
print(decode(m.generate(idx, max_new_tokens=400)[0].tolist()))
