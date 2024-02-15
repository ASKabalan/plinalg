import jax
import os

current_slurm = os.environ["SLURM_PROCID"]
print(f"{current_slurm} : cuda visible dev is {os.environ['CUDA_VISIBLE_DEVICES']}")



local_device_number = 4
local_ids = list(range(local_device_number)) 
# 0 1 2 3 for task 0
# 4 5 6 7 for task 1
current_slurm = int(current_slurm)
#local_ids = [f"cuda(id={i + 4*current_slurm})" for i in local_ids] 
local_ids = [i + 4*current_slurm for i in local_ids]
print(f"Current SLURM is {current_slurm} and local_ids are {local_ids}")
str_ids = [str(i) for i in local_ids]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_ids)
print(f"{current_slurm} : cuda visible dev is {os.environ['CUDA_VISIBLE_DEVICES']}")

jax.distributed.initialize(local_device_ids = local_ids,process_id=current_slurm)

if jax.process_index() == 0:
    print(f"All devices are: {jax.devices()}")

print(f"Local devices are {jax.local_devices()}")

jax.distributed.shutdown()
