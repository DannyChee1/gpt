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
    max_steps: int = 50
    seed: int = 1337