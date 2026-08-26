import torch

BATCH_SIZE = 4
BLOCK_SIZE = 8
MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'