import jax.numpy as jnp
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding , PartitionSpec as P
from jax.scipy.sparse.linalg import cg
import re
import time

A = jnp.array([[4, 0, 1, 0],
              [0, 5, 0, 0],
              [1, 0, 3, 2],
              [0, 0, 2, 4]])


b = jnp.array([-1, -0.5, -1, 2])


#print(exit_code)    # 0 indicates successful convergence

# Test Normal execuction
start = time.time()
x, exit_code = cg(A, b, atol=1e-5)
elabsed = time.time() - start
print(f"All ok {jnp.allclose(A.dot(x), b)} it took {elabsed} seconds")

# Test JIT
jitted_cg = jax.jit(cg)
#warm start
x, exit_code = jitted_cg(A, b, atol=1e-5)
# begin testing
start = time.time()
x, exit_code = jitted_cg(A, b, atol=1e-5)
elabsed = time.time() - start
print(f"All ok for jitted {jnp.allclose(A.dot(x), b)} it took {elabsed} seconds")
# Test sharded
print(f"beging testing multi gpu single controller")
print(f"Number of devices {jax.device_count()} they are {jax.devices()}")

devices = mesh_utils.create_device_mesh((jax.device_count(),))
mesh = Mesh(devices, ('x',))
pspecs = P('x')
A_sharded = jax.device_put(A , NamedSharding(mesh,pspecs))
jax.debug.visualize_array_sharding(A_sharded)

hlo_graph = jax.jit(cg).lower(A_sharded,b).compile().runtime_executable().hlo_modules()[0].to_string()

print(f"The HLO graph is {hlo_graph}")

_PATTERN = '(dynamic-slice|all-gather)'
# Did cg know how to deal with sharded array or did it all-gather then scatter
print(f"All gather happend {re.search(_PATTERN, hlo_graph) is not None}")

start = time.time()
x, exit_code = jitted_cg(A_sharded, b, atol=1e-5)
elabsed = time.time() - start
print(f"All ok for sharded jitted {jnp.allclose(A_sharded.dot(x), b)} it took {elabsed} seconds")

# Try to shard b the same way

b_sharded = jax.device_put(b , NamedSharding(mesh,pspecs))
jax.debug.visualize_array_sharding(b_sharded)
hlo_graph = jax.jit(cg).lower(A_sharded,b_sharded).compile().runtime_executable().hlo_modules()[0].to_string()
print(f"The HLO graph for two sharded arrays is {hlo_graph}")
# retry for A and B sharded
_PATTERN = '(dynamic-slice|all-gather)'
# Did cg know how to deal with sharded array or did it all-gather then scatter
print(f"All gather happend again {re.search(_PATTERN, hlo_graph) is not None}")

start = time.time()
x, exit_code = jitted_cg(A_sharded, b_sharded, atol=1e-5)
elabsed = time.time() - start
print(f"All ok for sharded jitted {jnp.allclose(A_sharded.dot(x), b_sharded)} it took {elabsed} seconds")


