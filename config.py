from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


@dataclass
class TrainConfig:
    data_path: str = "text.txt"
    batch_size: int = 4
    seq_len: int = 1024
    learning_rate: float = 3e-4
    max_lr: float = 6e-4
    min_lr: float = 6e-5 # max_lr * 0.1
    warmup_steps: int = 10
    max_steps: int = 30
    seed: int = 1337
    weight_decay: float = 0.1
