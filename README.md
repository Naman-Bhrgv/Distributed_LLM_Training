pytorch_ddp.py ->
minimal end-to-end PyTorch Distributed Data Parallel (DDP) training pipeline

Process -
Initialize DDP <br>
Create identical model on both GPUs <br>
Each GPU gets different data <br>
Forward Pass <br>
Compute Local Loss <br>
Backward Pass <br>
DDP AllReduce (Gradient Sync) <br>
Optimizer Step


Execution Loop- <br>
torchrun <br>
    │ 
    ▼
Launch 2 processes <br>
    │
    ▼
Initialize communication (init_process_group) <br>
    │
    ▼
Assign one GPU to each process <br>
    │
    ▼
Create identical model on each GPU <br>
    │
    ▼
Wrap with DDP <br>
    │
    ▼
Split dataset using DistributedSampler <br>
    │
    ▼
For each epoch: <br>
    │
    ├── Shuffle dataset (set_epoch) <br>
    ├── Load a unique batch on each GPU <br>
    ├── Forward pass (independent) <br>
    ├── Compute local loss <br>
    ├── Backward pass <br>
    ├── DDP automatically AllReduces gradients <br>
    ├── Clip gradients <br>
    ├── Optimizer updates weights identically on all GPUs <br>
    └── Accumulate loss for logging <br>
    │
    ▼
Reduce losses to Rank 0 <br>
    │
    ▼
Rank 0 prints metrics <br>
    │
    ▼
Rank 0 saves checkpoint <br>
    │
    ▼
Destroy process group <br>
