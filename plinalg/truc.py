import jax
from plinalg.hermitian import phermitian
from jax.lib import xla_bridge
from jax import jit,jacfwd,jacrev
import numpy as np
import json

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

print(xla_bridge.get_backend().platform)
#Create a complex 2D array
x = jax.random.normal(jax.random.PRNGKey(0), (2, 3)) + \
    1j * jax.random.normal(jax.random.PRNGKey(1), (2, 3))

origin_func = lambda x: x.conj().T

print(x)
print("*"*77)
jax_res = origin_func(x)
print(jax_res)
# Apply the phermitian function
print("*"*77)
result = phermitian(x)
print(result)
print(f"cuda is same as jax res {np.allclose(jax_res,result)}")
print("*"*77)
jitted_res = jit(phermitian)(x)
print(jitted_res)
print(f"jitted cuda is same as jax res {np.allclose(jitted_res,jax_res)}")

#Only real array for gradients
x = jax.random.normal(jax.random.PRNGKey(0), (10, 3))

x_batched = x.reshape(5 , x.shape[0]//5, *x.shape[1:])
print(f"x_batched shape {x_batched.shape}")
vmapped = jax.vmap(phermitian)(x_batched)
vmapped_original = jax.vmap(origin_func)(x_batched)
print(f"vmapped shape {vmapped.shape}")
print(f"vmapped origin shape {vmapped_original.shape}")
print(f"vmapped cuda is same as jax res vmapped {np.allclose(vmapped,vmapped_original)}")

x = jax.random.normal(jax.random.PRNGKey(0), (2, 3))

grad_func = jacfwd(phermitian)(x)
grad_origin = jacfwd(origin_func)(x)

print(f"grad cuda is same as jax grad {np.allclose(grad_func,grad_origin)}")

x = jax.random.normal(jax.random.PRNGKey(0), (2, 3)) + \
    1j * jax.random.normal(jax.random.PRNGKey(1), (2, 3))

rev_grad_func = jacrev(phermitian,holomorphic=True)(x)
rev_grad_origin = jacrev(origin_func,holomorphic=True)(x)
print(f"rev grad cuda is same as jax rev grad {np.allclose(rev_grad_func,rev_grad_origin)}")



