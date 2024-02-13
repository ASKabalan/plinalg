# plinalg

a demo for SPMD Linear Algebra using JAX

## Overview
plinalg is a library that provides a hermitian operator in JAX exported from cuda code.

Hermitian operator is simply a Conjugate transpose 

The Hermitian operator, denoted by $H$, is an important concept in linear algebra. It is defined as the conjugate transpose of a matrix or a vector. 

For a matrix $A$, the Hermitian operator is represented as $A^H$ or $A^\dagger$, where each element of $A$ is replaced by its complex conjugate and then the matrix is transposed.

The Hermitian operator has an important property that it is its own transpose (inverse) usefull when declaring a reverse mode gradient in JAX

## Installation
To install plinalg, you can use pip:

```shell
pip install plinalg@git+https://github.com/ASKabalan/plinalg
```

## Usage

check [test_primitive](tests/test_primitive.py)

You can call it as any other JAX function 


```python
import jax
from plinalg import hermitian

x = jax.random.normal(jax.random.PRNGKey(0), (2, 3)) + \
    1j * jax.random.normal(jax.random.PRNGKey(1), (2, 3))

result = hermitian(x)
```

It is also differentiable in forward and backward mode :

```python
from jax import jacrev,jacfwd

jacced_herm = jacfwd(hermitian,holomorphic=True)
hessian_herm = jacrev(jacced_herm ,holomorphic=True)

diff = jacced_herm(x)
diff_diff = hessian_herm(x)

```

It is also batchabble :

```python
from jax import vmap

xs = jax.random.normal(jax.random.PRNGKey(0), (10, 3)) + \
    1j * jax.random.normal(jax.random.PRNGKey(1), (10, 3))

x_batched = xs.reshape(5 , xs.shape[0]//5, *xs.shape[1:])

vmapped = vmap(hermitian)(x_batched)
assert vmapped.shape == (5 , 2 , 3)
```

And most importantly batchabble on multiple devices in a single controller multi device setup : 
check [test_spmd_lowering](tests/test_spmd_lowering.py)

```python
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding

P = PartitionSpec

x = jax.random.normal(jax.random.PRNGKey(0), (16, 3)) + \
    1j * jax.random.normal(jax.random.PRNGKey(1), (16, 3))

devices = mesh_utils.create_device_mesh((8,))
mesh = Mesh(devices, ('x',))
# Make a sharded device array
SDA = jax.device_put(x, NamedSharding(mesh, P('x',)))
SDA = SDA.reshape(jax.device_count(), SDA.shape[0] // jax.device_count(), *SDA.shape[1:])
# we can also use jax.local_device_count() instead since in this setup they are the same

@partial(shard_map,mesh=mesh,in_specs=P("x", None, None),out_specs=P("x", None, None),check_rep=False)
def phermitian(x):
    return hermitian(x)

with mesh:
    jitted = jit(phermitian)
    out = jitted(SDA)

# The XLA graph does no all gather : 

HloModule pjit_phermitian, is_scheduled=true, entry_computation_layout={(c64[1,2,3]{2,1,0})->c64[1,3,2]{2,1,0}}, num_partitions=8, frontend_attributes={fingerprint_before_lhs="4564bf20f1db9a2290fee42597b1f4c0"}

ENTRY %main.12_spmd (param: c64[1,2,3]) -> c64[1,3,2] {
  %param = c64[1,2,3]{2,1,0} parameter(0), sharding={devices=[8,1,1]<=[8]}, metadata={op_name="pjit(phermitian)/jit(main)/shard_map[mesh=Mesh(\'x\': 8) in_names=({0: (\'x\',)},) out_names=({0: (\'x\',)},) check_rep=False rewrite=False auto=frozenset()]" source_file="/gpfsdswork/projects/rech/.../Projects/plinalg/tests/test_spmd_lowering.py" source_line=135}
  ROOT %custom-call.1 = c64[1,3,2]{2,1,0} custom-call(c64[1,2,3]{2,1,0} %param), custom_call_target="hermitian_operator", operand_layout_constraints={c64[1,2,3]{2,1,0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(phermitian)/jit(main)/jit(shmap_body)/phermitian" source_file="/gpfsdswork/projects/rech/.../Projects/plinalg/plinalg/hermitian.py" source_line=29}, backend_config="\001\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\003\000\000\000\000\000\000\000"
}

# If we only jit

...
%all-gather-start = (c64[2,3]{1,0}, c64[16,3]{1,0}) all-gather-start(c64[2,3]{1,0} %param), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(hermitian)/jit(main)/phermitian" source_file="/gpfsdswork/projects/rech/.../Projects/plinalg/plinalg/hermitian.py" source_line=29}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false}}
%all-gather-done = c64[16,3]{1,0} all-gather-done((c64[2,3]{1,0}, c64[16,3]{1,0}) %all-gather-start), metadata={op_name="pjit(hermitian)/jit(main)/phermitian" source_file="/gpfsdswork/projects/rech/.../Projects/plinalg/plinalg/hermitian.py" source_line=29}
...

```


it can also work for a multi controller setup 

```python
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map

P = PartitionSpec

# We have 8 (2 , 3) shaped slices so (16 , 3) GDA just like the single controller case
x = jax.random.normal(jax.random.PRNGKey(0), (2, 3)) + \
    1j * jax.random.normal(jax.random.PRNGKey(1), (2, 3))

devices = mesh_utils.create_device_mesh((8,))
mesh = Mesh(devices, ('x',))
pspecs = P('x')
# Make global device array
GDA = multihost_utils.host_local_array_to_global_array(x,mesh,pspecs)
GDA = GDA.reshape(jax.device_count(), GDA.shape[0] // jax.device_count(), *GDA.shape[1:])

@partial(shard_map,mesh=mesh,in_specs=P("x", None, None),out_specs=P("x", None, None),check_rep=False)
def phermitian(x):
    return hermitian(x)

with mesh:
    jitted = jit(phermitian)
    out = jitted(SDA)

```
