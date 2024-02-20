# Using the slurms

make sure that you have correct python venv

For an interactive use

```bash
salloc --account=nih@v100 --nodes=2  --ntasks-per-node=4 --gres=gpu:4 -C v100-16g --hint=nomultithread --qos=qos_gpu-dev   
```

or for a long dev session

```bash
salloc --account=nih@v100 --nodes=2  --ntasks-per-node=4 --gres=gpu:4 -C v100-16g --hint=nomultithread --time=04:00:00  
```
or 

```bash
srun --pty --time=04:00:00   --account=nih@v100 --nodes=2  --ntasks-per-node=4 --gres=gpu:4 -C v100-16g --hint=nomultithread  bash 
```

To install dependencies


```bash
module load cuda/11.8.0  cudnn/8.9.7.29-cuda cmake nvidia-compilers/23.9
export MODULEPATH=$NVHPC/modulefiles:$MODULEPATH
module load nvhpc-hpcx-cuda11/23.9

module load python/3.10.4
python -m venv venv

pip cache purge
source venv/bin/activate
# Installing mpi4py
CFLAGS=-noswitcherror pip install --no-cache-dir mpi4py
# Installing jax
pip install --no-cache-dir --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Once it is installed to load modules

```bash
module load cuda/11.8.0  cudnn/8.9.7.29-cuda cmake nvidia-compilers/23.9
export MODULEPATH=$NVHPC/modulefiles:$MODULEPATH
module load nvhpc-hpcx-cuda11/23.9
module load python/3.10.4 && conda deactivate
source venv/bin/activate
```