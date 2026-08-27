import torch

batch_size = 4
block_size = 8
max_iters = 3000
eval_interval = 300
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 32
