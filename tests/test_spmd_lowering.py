
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.pjit import pjit
from jax.experimental import mesh_utils,multihost_utils
import pytest
from plinalg.hermitian import hermitian
import jax
import jax.numpy as jnp
from jax.experimental.maps import xmap
from jax.experimental.shard_map import shard_map
from functools import partial
from jax import jit
import re

jax.distributed.initialize()

if jax.process_index() == 0:
    print(f"Total number of devices: {jax.device_count()}")
    print(f"Global devices names: {jax.devices()}")

print(f"Process {jax.process_index()} has {jax.local_device_count()} devices: {jax.local_devices()}")

P = PartitionSpec


x = jax.random.normal(jax.random.PRNGKey(0), (2, 3)) + \
    1j * jax.random.normal(jax.random.PRNGKey(1), (2, 3))

devices = mesh_utils.create_device_mesh((jax.device_count(),))
mesh = Mesh(devices, ('x',))
pspecs = P('x')
global_array = multihost_utils.host_local_array_to_global_array(x,mesh,pspecs)
global_resh = global_array.reshape(jax.device_count(), global_array.shape[0] // jax.device_count(), *global_array.shape[1:])

# Pattern to detect all-gather or dynamic-slice in the generated HLO
_PATTERN = '(dynamic-slice|all-gather)'

def test_global_shape():

    local_shape = x.shape
    expected_global_shape = (jax.device_count() * local_shape[0] , *local_shape[1:])
    assert global_array.shape == expected_global_shape
    expected_global_resh_shape = (jax.device_count(), *x.shape)
    assert global_resh.shape == expected_global_resh_shape

def test_all_gather():

    pjitted = pjit(
        hermitian,
        # Shard x by batch dimension and replicate weight on all devices.
        in_shardings=P("x", None),
        # Shard the output by batch dimension.
        out_shardings=P("x", None),
    )

    with mesh:
        hlo_graph = pjitted.lower(global_array).compile().runtime_executable().hlo_modules()[0].to_string()
        out = pjitted(global_array)
    
    if jax.process_index() == 0:
        print(f"Printing HLO Graph that was only pjitted")
        print(hlo_graph)

        #check that hlo_graph did an all-gather followed by a dynamic-slice
        assert(re.search(_PATTERN, hlo_graph) is not None)

        # Don't know how to compare in multi controller setup
        #assert jnp.allclose(ref, out, atol=1e-2, rtol=1e-2)


def test_xmap_lowering():


    def phermitian(x, *, device_count):
        reshaped = x.reshape(device_count, x.shape[0] // device_count, *x.shape[1:])
        xmapped = xmap(
            hermitian,
            in_axes=("x", None, None),
            out_axes=("x", None, None),
            axis_resources={"x": "x"},
        )
        reshaped_out = xmapped(reshaped)
        return reshaped_out.reshape(x.shape)

    with mesh:

        pjitted = pjit(
            partial(phermitian, device_count=jax.device_count()),
            # Shard x by batch dimension and replicate weight on all devices.
            in_shardings=P("x", None),
            # Shard the output by batch dimension.
            out_shardings=P("x", None),
        )
        hlo_graph = pjitted.lower(global_array).compile().runtime_executable().hlo_modules()[0].to_string()
        out = pjitted(global_array)

    if jax.process_index() == 0:
        print(f"Printing HLO Graph that was xmapped and pjitted")
        print(hlo_graph)

        #check that hlo_graph did not do an all-gather followed by a dynamic-slice
        # assert(re.search(_PATTERN, hlo_graph) is None)

        # Don't know yet how to compare in multi controller setup (maybe vmap in first device?)
        #assert jnp.allclose(ref, out, atol=1e-2, rtol=1e-2)

def test_shard_mapped_lowering():

    @partial(shard_map,mesh=mesh,in_specs=P("x", None, None),out_specs=P("x", None, None))
    def phermitian(x):
        return hermitian(x)
    
    jitted = jit(phermitian,in_shardings=P("x", None, None),out_shardings=P("x", None, None))

    hlo_graph = jitted.lower(global_resh).compile().runtime_executable().hlo_modules()[0].to_string()
    out = phermitian(global_resh)

    if jax.process_index() == 0:
        print(f"Printing HLO Graph that was shard_mapped and jitted")
        print(hlo_graph)

        #check that hlo_graph did not do an all-gather followed by a dynamic-slice
        # assert(re.search(_PATTERN, hlo_graph) is None)

        # Don't know yet how to compare in multi controller setup (maybe vmap in first device?)
        #assert jnp.allclose(ref, out, atol=1e-2, rtol=1e-2)



from jax.sharding import NamedSharding
from jax.experimental.custom_partitioning import custom_partitioning

@pytest.mark.skip(reason="I am not sure yet what I am doing")
def test_custom_partitionning_lowering():
    # For an N-D input, keeps sharding along the first N-1 dimensions
    # but replicate along the last dimension
    def supported_sharding(sharding, shape):
        rank = len(shape.shape)
        max_shared_dims = min(len(sharding.spec), rank-1)
        names = tuple(sharding.spec[:max_shared_dims]) + tuple(None for _ in range(rank - max_shared_dims))
        return NamedSharding(sharding.mesh, P(*names))

    def partition(mesh, arg_shapes, result_shape):
        result_shardings = jax.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, fft,               supported_sharding(arg_shardings[0], arg_shapes[0]),               (supported_sharding(arg_shardings[0], arg_shapes[0]),)

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
        arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
        return supported_sharding(arg_shardings[0], arg_shapes[0])

    @custom_partitioning
    def phermitian(x):
        return hermitian(x)

    phermitian.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition)
    
    with mesh:
        pjit_phermitian = pjit(phermitian, in_shardings=P('x'), out_shardings=P('x'))
        pjit_hermitian    = pjit(hermitian,    in_shardings=P('x'), out_shardings=P('x'))
        print(pjit_phermitian(global_array))
        print(pjit_hermitian(global_array))
        # dynamic-slice or all-gather are not present in the HLO for my_fft, because x is a 2D array
        assert(re.search(_PATTERN, pjit_phermitian.lower(x).compile().runtime_executable().hlo_modules()[0].to_string()) is None)
        # dynamic-slice or all-gather are present in the HLO for fft
        assert(re.search(_PATTERN, pjit_hermitian.lower(x).compile().runtime_executable().hlo_modules()[0].to_string())    is not None)