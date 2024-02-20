
import jax
jax.distributed.initialize()

from jax.sharding import Mesh, PartitionSpec
from jax.experimental.pjit import pjit
from jax.experimental import mesh_utils,multihost_utils
import pytest
import jax.numpy as jnp
from jax.experimental.maps import xmap
from jax.experimental.shard_map import shard_map
from functools import partial
from jax import jit,vmap
import re
from plinalg.debug_tools import inspect_attr

from plinalg.hermitian import hermitian

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

# Compute the all-gathered array on one GPU for reference.
global_all_gathered = multihost_utils.process_allgather(global_array)
global_all_gathered = global_all_gathered.reshape(jax.device_count(), global_array.shape[0] // jax.device_count(), *global_array.shape[1:])

if jax.process_index() == 0:

    ref = vmap(hermitian)(global_all_gathered)


multihost_utils.sync_global_devices("For Printing")

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
        out_shardings=P(None, "x"),#Notice that it was transposed because it was hermitianned on the all gathered matrix rather than on the batches
    )

    with mesh:
        hlo_graph = pjitted.lower(global_array).compile().runtime_executable().hlo_modules()[0].to_string()
        out = pjitted(global_array)
    
    if jax.process_index() == 0:
        print(f"Printing HLO Graph that was only pjitted")
        print(hlo_graph)

        #check that hlo_graph did an all-gather followed by a dynamic-slice
        assert(re.search(_PATTERN, hlo_graph) is not None)

        # Don't know how to compare in multi controller setup (Maybe use vmap in first device?)
        #assert jnp.allclose(ref, out, atol=1e-2, rtol=1e-2)

multihost_utils.sync_global_devices("For Printing")


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
        assert(re.search(_PATTERN, hlo_graph) is None)

        # Don't know yet how to compare in multi controller setup (maybe vmap in first device?)
        #assert jnp.allclose(ref, out, atol=1e-2, rtol=1e-2)

multihost_utils.sync_global_devices("For Printing")


def test_shard_mapped_lowering():

    @partial(shard_map,mesh=mesh,in_specs=P("x", None, None),out_specs=P("x", None, None),check_rep=False)
    def phermitian(x):
        return hermitian(x)
    
    with mesh:
        jitted = pjit(           
                phermitian,
                # Shard x by batch dimension and replicate weight on all devices.
                in_shardings=P("x", None, None),
                # Shard the output by batch dimension.
                out_shardings=P("x", None, None),
        )

        hlo_graph = jitted.lower(global_resh).compile().runtime_executable().hlo_modules()[0].to_string()
        out = phermitian(global_resh)

    if jax.process_index() == 0:
        print(f"Printing HLO Graph that was shard_mapped and jitted")
        print(hlo_graph)

        #check that hlo_graph did not do an all-gather followed by a dynamic-slice
        assert(re.search(_PATTERN, hlo_graph) is None)

        # Don't know yet how to compare in multi controller setup (maybe vmap in first device?)
        #assert jnp.allclose(ref, out, atol=1e-2, rtol=1e-2)

multihost_utils.sync_global_devices("For Printing")

def test_shard_mapped_lowering_with_jit():

    @partial(shard_map,mesh=mesh,in_specs=P("x", None, None),out_specs=P("x", None, None),check_rep=False)
    def phermitian(x):
        return hermitian(x)
    
    with mesh:
        jitted = jit(phermitian)

        hlo_graph = jitted.lower(global_resh).compile().runtime_executable().hlo_modules()[0].to_string()
        out = phermitian(global_resh)

    if jax.process_index() == 0:
        print(f"Printing HLO Graph that was shard_mapped and jitted")
        print(hlo_graph)

        #check that hlo_graph did not do an all-gather followed by a dynamic-slice
        assert(re.search(_PATTERN, hlo_graph) is None)

        # Don't know yet how to compare in multi controller setup (maybe vmap in first device?)
        #assert jnp.allclose(ref, out, atol=1e-2, rtol=1e-2)

multihost_utils.sync_global_devices("For Printing")

from jax.sharding import NamedSharding
from jax.experimental.custom_partitioning import custom_partitioning
TRACING_STRING = 'TRACING-WKN3'
#@pytest.mark.skip(reason="I am not sure yet what I am doing")
def test_custom_partitionning():
    # For an N-D input, keeps sharding along the first N-1 dimensions
    # but replicate along the last dimension


    def propagate_user_sharding(mesh, user_shape):

        inspect_attr(mesh,"mesh" , f"{TRACING_STRING} propagate_user_sharding")
        inspect_attr(mesh,"user_shape" , f"{TRACING_STRING} user_shape")
        user_sharding = jax.tree_map(lambda x: x.sharding, user_shape)
        inspect_attr(mesh,"user_sharding" , f"{TRACING_STRING} user_sharding")
        return user_sharding


    def supported_sharding(sharding, shape):

        inspect_attr(sharding,"Sharding" , f"{TRACING_STRING} supported_sharding")
        inspect_attr(shape,"Shape" , f"{TRACING_STRING} supported_sharding")

        rank = len(shape.shape)
        max_shared_dims = min(len(sharding.spec), rank-1)
        names = tuple(sharding.spec[:max_shared_dims]) + tuple(None for _ in range(rank - max_shared_dims))
        return NamedSharding(sharding.mesh, P(*names))


    def partition(mesh, arg_shapes, result_shape):
        
        """
        Tells XLA how to partition the primitive

        Args:
            mesh (Mesh): The contextual mesh 

            arg_shapes (tuple): A tuple of ShapeDtypeStruct that contains the shape and the sharding of each input operand

            result_shape (ShapeDtypeStruct) : a ShapeDtypeStruct reprsenting a single output

        Returns:
            Mesh (Mesh) : The mesh. 

            function: The lowered function, to allow the user to redefine how the primitive is called in a context of a specific sharding 
                    
            result_sharding (XLACompatibleSharding): The sharding result for example a NamedSharding. 

            arg_shardings (tuple): a tuple of all XLACompatibleSharding of the input operands
        """
        
        inspect_attr(mesh,"Mesh" , f"{TRACING_STRING} partition")
        inspect_attr(arg_shapes,"Arg Shapes" , f"{TRACING_STRING} partial")
        inspect_attr(result_shape,"Result Shape" , f"{TRACING_STRING} partial")

        result_shardings = jax.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)

        inspect_attr(arg_shardings,"Arg Shardings" , f"{TRACING_STRING} partial")
        inspect_attr(result_shardings,"Result Shardings" , f"{TRACING_STRING} partial")
        
        return mesh, hermitian,\
            supported_sharding(arg_shardings[0], arg_shapes[0]),\
                  (supported_sharding(arg_shardings[0], arg_shapes[0]),)
    

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
        """
        Tell XLA how to infer the sharding of the output from the input sharding.

        Args:
            mesh (Mesh): The contextual mesh 

            arg_shapes (tuple): A tuple of ShapeDtypeStruct that contains the shape and the sharding of each input operand

            result_shape (ShapedArray) : a single ShapedArray reprsenting a single output without the sharding information

        Returns:
                
            result_sharding (XLACompatibleSharding): The sharding result for example a NamedSharding. 

        """
        inspect_attr(mesh,"Mesh" , f"{TRACING_STRING} infer_sharding_from_operands")
        inspect_attr(arg_shapes,"Arg Shapes" , f"{TRACING_STRING} infer_sharding_from_operands")
        inspect_attr(result_shape,"Result Shape" , f"{TRACING_STRING} infer_sharding_from_operands")

        arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)

        inspect_attr(arg_shardings,"Arg Shardings" , f"{TRACING_STRING} infer_sharding_from_operands")
        inspect_attr(arg_shardings[0],"Arg Shardings 0" , f"{TRACING_STRING} infer_sharding_from_operands")
        inspect_attr(arg_shapes[0],"Arg Shapes 0" , f"{TRACING_STRING} infer_sharding_from_operands")


        return supported_sharding(arg_shardings[0], arg_shapes[0])

    @custom_partitioning
    def phermitian(x):
        return hermitian(x)

    phermitian.def_partition(
        propagate_user_sharding=propagate_user_sharding,
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition)
    
    with mesh:
        pjit_phermitian = pjit(phermitian, in_shardings=P("x", None, None), out_shardings=P("x", None, None))
        pjit_hermitian    = pjit(hermitian,    in_shardings=P("x", None, None), out_shardings=P("x", None, None))
        #print(pjit_phermitian(global_array))
        #print(pjit_hermitian(global_array))
        # dynamic-slice or all-gather are not present in the HLO for my_fft, because x is a 2D array
        #assert(re.search(_PATTERN, pjit_hermitian.lower(x).compile().runtime_executable().hlo_modules()[0].to_string()) is None)
        print(f"HLO graph hermitian {pjit_hermitian.lower(global_resh).compile().runtime_executable().hlo_modules()[0].to_string()}" )
        # dynamic-slice or all-gather are present in the HLO for fft
        #assert(re.search(_PATTERN, pjit_hermitian.lower(x).compile().runtime_executable().hlo_modules()[0].to_string())    is not None)
        print(f"HLO graph parallel hermitian {pjit_phermitian.lower(global_resh).compile().runtime_executable().hlo_modules()[0].to_string()}" )

multihost_utils.sync_global_devices("For Printing")

jax.distributed.shutdown()
