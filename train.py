import os
import torch
import time
import math

from config import GPTConfig, TrainConfig
from data import DataLoader
from model import GPT
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_lr(iter):
    if iter < train_config.warmup_steps:
        return train_config.max_lr * (iter + 1) / train_config.warmup_steps
    if iter > train_config.max_steps:
        return train_config.min_lr
    decay_ratio = (iter - train_config.warmup_steps) / (train_config.max_steps - train_config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (train_config.max_lr - train_config.min_lr)

def setup_distributed():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA is not available"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_world_size = 1
        ddp_local_rank = 0
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"Using device: {device}")
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device

ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_distributed()
train_config = TrainConfig()

torch.manual_seed(train_config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(train_config.seed)


total_batch_size = int(os.environ.get("TOTAL_BATCH_SIZE", 524288)) # 2^19, ~0.5M in tokens; override for quick local runs
tokens_per_micro_batch = train_config.batch_size * train_config.seq_len * ddp_world_size
assert total_batch_size % tokens_per_micro_batch == 0, "total_batch_size must be divisible by B * T"
grad_accumulation_steps = total_batch_size // tokens_per_micro_batch

if master_process:
    print(f"Using total_batch_size: {total_batch_size} and grad_accumulation_steps: {grad_accumulation_steps}")

data = DataLoader(B=train_config.batch_size, T=train_config.seq_len, data_path=train_config.data_path, process_rank=ddp_rank, num_processes=ddp_world_size)

torch.set_float32_matmul_precision("high") # use TF32 instead of FP32 for faster computation

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model
optimizer = raw_model.configure_optimizers(train_config, device) # torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, betas=(0.9, 0.95), eps=1e-8)
for iter in range(train_config.max_steps):
    t0 = time.time()
    optimizer.zero_grad()

    total_loss = 0.0
    for step in range(grad_accumulation_steps):
        x, y = data.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accumulation_steps
        total_loss += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (step == grad_accumulation_steps - 1) # sync gradients at the end of the micro batch
        loss.backward()

    if ddp:
        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_config.batch_size * train_config.seq_len * ddp_world_size * grad_accumulation_steps
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {iter} loss: {total_loss.item():.6f} time: {dt*1000:.2f}ms tokens/sec: {tokens_per_sec:.2f} lr: {lr:.2e} norm: {norm.item():.2e}")

if ddp:
    destroy_process_group()