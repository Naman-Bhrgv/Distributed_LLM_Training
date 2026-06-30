ddp-pytorch.ipynb ->
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


DeepSpeed-

Both GPUs: forward pass (model_engine(x)) <br>
    ↓ <br>
Both GPUs: compute local loss and gradients (model_engine.backward(loss)) <br>
    ↓ <br>
ReduceScatter: sum gradients across GPUs, then split <br>
    GPU 0 keeps: gradient shard for its params <br>
    GPU 1 keeps: gradient shard for its params <br>
    ↓ <br>
Both GPUs: clip + optimizer step on their own shard (model_engine.step()) <br>
    ↓ <br>
AllGather: share updated params so everyone has full model <br>
    ↓ <br>
Next forward pass — both GPUs have identical weights, ready to go <br>
<br>
very call to model_engine.backward() and model_engine.step() now automatically uses ZeRO's sharded communication instead of standard AllReduce.

