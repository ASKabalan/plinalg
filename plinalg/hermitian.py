from plinalglib import _pHermitian

import jax
import jax.numpy as jnp
from jax.interpreters import mlir
from jax.lib import xla_client
from jax import custom_jvp
from jax.interpreters import xla, ad, batching
from jax.core import Primitive, ShapedArray
from jax._src.numpy.util import promote_dtypes_complex, promote_dtypes_inexact

import jaxlib.mlir.ir as ir
from jaxlib.hlo_helpers import custom_call
import json
from functools import partial

def get_attribute_values(obj):
    attribute_values = {}
    attributes = dir(obj)
    for attr in attributes:
        try:
            attribute_values[attr] = str(getattr(obj, attr))
        except Exception as e:
            attribute_values[attr] = str(e)
    return attribute_values

def pretty_print(attributes_values):
    print(json.dumps(attributes_values, indent=4))

def inspect_attr(var, var_name, name):
    print("{0}{2} - {3}{1}".format("*"*77,"*"*77,name,var_name))
    print(f"{name} => {var_name} : {var} \n")
    print(f"{name} => {var_name} Type : {type(var)} \n")
    pretty_print(get_attribute_values(var))
    print("\n")

####################
# Declare Primitive #
####################

phermitian_p = Primitive('phermitian')
phermitian_p.multiple_results = False

phermitian_p.def_impl(partial(xla.apply_primitive,phermitian_p))

def hermitian(x) :
    (x,) = promote_dtypes_complex(x)
    return phermitian_p.bind(x)

####################
# Lowering to MLIR #
####################

for _name, _value in _pHermitian.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

def _phermitation_lowering(ctx,x):

    inspect_attr(ctx,"Context","_phermitation_lowering")
    inspect_attr(x,"Operand","_phermitation_lowering")

    """Lower the Hermitian operator to CUDA via MLIR, ensuring output is complex and transposed."""
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    
    dims = x_type.shape
    batch_size = dims[0] if len(dims) > 2 else 1
    inspect_attr(batch_size, "batch_size", "phermitation_lowering")

    # Adjust for the transposed shape of the output.
    if len(dims) == 2:
        result_shape = (x_shape[-1], x_shape[-2])  # For 2D input
    elif len(dims) == 3:
        result_shape = (batch_size, x_shape[-1], x_shape[-2])  # For 3D input

    result_type = ir.RankedTensorType.get(result_shape, x_type.element_type)

    # Create the descriptor with batch size, matrix length (m), and matrix width (n).
    m, n = x_shape[-2], x_shape[-1]  # Assuming the last two dimensions are matrix dimensions.
    opaque = _pHermitian.build_hermitian_descriptor(batch_size, m, n)

    # The custom call to the GPU operation.
    out = custom_call(
        b'hermitian_operator',
        result_types=[result_type],
        operands=[x],
        backend_config=opaque,
        operand_layouts=default_layouts(x_shape),
        result_layouts=default_layouts(result_shape),
    ).results

    return out

mlir.register_lowering(
    phermitian_p,
    _phermitation_lowering,
    platform="gpu",
)

#######################
# Abstract evaluation #
#######################
from jax import dtypes

def phermitian_abstract_eval(x):

    if x.ndim < 2:
        raise ValueError("Input to phermitian must have at least 2 dimensions")
    if x.ndim > 3:
        raise ValueError("Only 2D arrays or Batched 2D arrays are supported for now")

    # Assuming we want to ensure the output is always complex,
    # we check the input dtype and promote to complex if necessary.
    # This mirrors how numpy would handle dtype promotion for complex operations.
    input_dtype = dtypes.canonicalize_dtype(x.dtype)
    if not dtypes.issubdtype(input_dtype, jnp.complexfloating):
        output_dtype = dtypes.complex64 if dtypes.finfo(input_dtype).bits <= 32 else dtypes.complex128
    else:
        output_dtype = input_dtype

    # Handle 2D and 3D arrays differently
    if x.ndim == 2:
        axes_order = (1, 0)  # Transpose 2D arrays
        transpose_shape = tuple(x.shape[i] for i in axes_order)
    elif x.ndim == 3:
        transpose_shape = (x.shape[0], x.shape[2] , x.shape[1])

    return ShapedArray(transpose_shape, output_dtype)


phermitian_p.def_abstract_eval(phermitian_abstract_eval)

#######################################
# Top-level interface Automatic Diff #
#######################################

def _phermitian_transpose_rule(t, x):
    return (hermitian(cotangent),)  

ad.deflinear2(phermitian_p, _phermitian_transpose_rule)

#######################################
# Batching #
#######################################


def phermitian_batching_rule(batched_args, batch_dims):
    (x,), (x_bdim,) = batched_args, batch_dims
    if x_bdim is None:
        return hermitian(x), None
    else:
        # Move the batch dimension to the front if it's not already there
        x = jnp.moveaxis(x, x_bdim, 0)
        return hermitian(x), 0

batching.primitive_batchers[phermitian_p] = phermitian_batching_rule

######################
# SPMD Defintion     #
######################

from jax.experimental.maps import xmap
jax.config.update("experimental_xmap_spmd_lowering", True)
jax.config.update("experimental_xmap_spmd_lowering_manual", True)

def phermitian(x, *, device_count):
    reshaped = x.reshape(device_count, x.shape[0] // device_count, *x.shape[1:])
    xmapped = xmap(
        rms_norm,
        in_axes=("x", None, None, None),
        out_axes=("x", None, None, None),
        axis_resources={"x": "x"},
    )
    reshaped_out = xmapped(reshaped, weight)
    return reshaped_out.reshape(x.shape)



