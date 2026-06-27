"""
Minimal PyTorch DDP training script — 2 GPUs
Launch with:
    torchrun --nproc_per_node=2 train_ddp.py

Requirements:
    pip install torch
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


# ── 1. A simple model ────────────────────────────────────────────────────────

class TinyTransformer(nn.Module):
    """Small transformer-like model just to exercise DDP meaningfully."""
    def __init__(self, vocab_size=1000, hidden=256, num_layers=4, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads,
            dim_feedforward=hidden * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.embed(x)               # [B, seq, hidden]
        x = self.transformer(x)         # [B, seq, hidden]
        return self.head(x)             # [B, seq, vocab_size]


# ── 2. Dummy dataset ─────────────────────────────────────────────────────────

def make_dataset(num_samples=1024, seq_len=64, vocab_size=1000):
    """Random token sequences — replace with your real dataset."""
    x = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, vocab_size, (num_samples, seq_len))
    return TensorDataset(x, y)


# ── 3. Training loop ─────────────────────────────────────────────────────────

def train(rank, world_size, epochs=3):
    # ── Init process group ──────────────────────────────────────────────────
    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE automatically
    dist.init_process_group(backend="nccl")   # used to initialize communication between multiple processes
    torch.cuda.set_device(rank)  #tells PyTorch which GPU the current process should use.
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"Training on {world_size} GPUs\n")

    # ── Model ───────────────────────────────────────────────────────────────
    model = TinyTransformer().to(device)
    # DDP broadcasts rank-0 weights to all GPUs at construction — all start identical
    model = DDP(model, device_ids=[rank])

    # ── Data ────────────────────────────────────────────────────────────────
    dataset = make_dataset()
    # DistributedSampler ensures no two GPUs see the same sample in one epoch
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) #ensure that each GPU processes a unique portion of the dataset instead of all GPUs processing the entire dataset
    loader  = DataLoader(dataset, batch_size=32, sampler=sampler, pin_memory=True) # creates a DataLoader that loads data efficiently and feeds it to your model during training

    # ── Optimizer & loss ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # ── Epoch loop ──────────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()

        # IMPORTANT: set_epoch shuffles differently each epoch.
        # Without this, every epoch uses the same data order.
        sampler.set_epoch(epoch)

        total_loss = torch.tensor(0.0, device=device)
        num_batches = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Forward — runs independently on each GPU
            logits = model(x)                              # [B, seq, vocab]

            # Loss — each GPU computes loss on its own mini-batch
            loss = criterion(logits.view(-1, 1000), y.view(-1))  #computes the training loss between the model's predictions (logits) and the true labels (y)

            # Backward — DDP automatically triggers AllReduce for gradient sync
            # Gradients are averaged across both GPUs before optimizer.step()
            loss.backward()

            # Clip gradients AFTER AllReduce (after backward completes)
            # Clipping before would clip each GPU's local grads independently
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.detach() # adds the current batch's loss to a running total without keeping it in PyTorch's computation graph.
            #total_loss is only used for monitoring (e.g., computing the average loss), not for training the model. Therefore, there is no reason for PyTorch to track gradients through it.
            num_batches += 1

        # ── Logging (rank 0 only) ────────────────────────────────────────
        # Reduce loss across GPUs so rank 0 can print the true average
        dist.reduce(total_loss, dst=0, op=dist.ReduceOp.AVG)    #used in Distributed Data Parallel (DDP) to combine total_loss from all processes (GPUs) into a single value
        if rank == 0:
            avg_loss = total_loss.item() / num_batches
            print(f"Epoch {epoch+1}/{epochs}  |  loss: {avg_loss:.4f}")

    # ── Save checkpoint (rank 0 only) ────────────────────────────────────────
    # model.module gives the underlying model, not the DDP wrapper
    # Always save model.module.state_dict(), not model.state_dict()
    # model.state_dict() adds "module." prefix to every key which breaks loading
    if rank == 0:
        torch.save(model.module.state_dict(), "checkpoint.pt")
        print("\nCheckpoint saved to checkpoint.pt")

    dist.destroy_process_group()


# ── 4. Entry point ───────────────────────────────────────────────────────────

def main():
    # torchrun injects these env vars — you don't set them manually
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train(rank, world_size)


if __name__ == "__main__":
    main()
