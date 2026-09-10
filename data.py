import tiktoken
import torch


class DataLoader:
    def __init__(self, B, T, data_path="text.txt"):
        self.B = B
        self.T = T

        with open(data_path, "r") as f:
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
