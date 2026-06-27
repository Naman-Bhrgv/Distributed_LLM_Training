pytorch_ddp.py ->
minimal end-to-end PyTorch Distributed Data Parallel (DDP) training pipeline

Process -
Initialize DDP
Create identical model on both GPUs
Each GPU gets different data
Forward Pass
Compute Local Loss
Backward Pass
DDP AllReduce (Gradient Sync)
Optimizer Step

torchrun
    │
    ▼
Launch 2 processes
    │
    ▼
Initialize communication (init_process_group)
    │
    ▼
Assign one GPU to each process
    │
    ▼
Create identical model on each GPU
    │
    ▼
Wrap with DDP
    │
    ▼
Split dataset using DistributedSampler
    │
    ▼
For each epoch:
    │
    ├── Shuffle dataset (set_epoch)
    ├── Load a unique batch on each GPU
    ├── Forward pass (independent)
    ├── Compute local loss
    ├── Backward pass
    ├── DDP automatically AllReduces gradients
    ├── Clip gradients
    ├── Optimizer updates weights identically on all GPUs
    └── Accumulate loss for logging
    │
    ▼
Reduce losses to Rank 0
    │
    ▼
Rank 0 prints metrics
    │
    ▼
Rank 0 saves checkpoint
    │
    ▼
Destroy process group
