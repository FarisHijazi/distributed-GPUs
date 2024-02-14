# distributed-GPUs
learning about distributed GPUs through Ray, SLURM, PyTorch, and other tools


## Raw PyTorch concepts

Resource: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

The gist is:

```python
...
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
...


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    ...

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

```

## SLURM PyTorch concepts

```sh
pip install -r requirements.txt
python main.py
# lauch 2 gpus x 2 nodes (= 4 gpus)
srun -N2 -p gpu --gres gpu:2 python pytorch_slurm.py --dist-backend nccl --multiprocessing-distributed --dist-file dist_file
```

```python
```

## Ray concepts

## SLURM concepts

A workload manager to reserve resources (CPU, RAM, GPU, ...)

Cheat sheet:

![](https://rc-docs.northeastern.edu/en/latest/_images/slurm-cheatsheet-1.png)

At the end of the day, you're running a bash script that runs a python script.

It will look something like this:

```sh
#!/bin/bash
#SBATCH --job-name=your-job-name
#SBATCH --partition=gpu
#SBATCH --time=72:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=p40&gmem24G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --chdir=/scratch/shared/beegfs/your_dir/
#SBATCH --output=/scratch/shared/beegfs/your_dir/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv

### the command to run
srun python main.py --net resnet18 \
--lr 1e-3 --epochs 50 --other_args
```

## Resources

- https://github.com/statgen/SLURM-examples
- 

