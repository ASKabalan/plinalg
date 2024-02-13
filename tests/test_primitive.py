import pytest
import jax
from plinalg.hermitian import hermitian
from jax import jit,jacfwd,jacrev
import numpy as np

x = jax.random.normal(jax.random.PRNGKey(0), (2, 3)) + \
    1j * jax.random.normal(jax.random.PRNGKey(1), (2, 3))

x_batched = jax.random.normal(jax.random.PRNGKey(0), (10, 3))


def origin_func(x):
    return x.conj().T

def test_hermitian():
    
    jax_res = origin_func(x)
    result = hermitian(x)
    assert np.allclose(jax_res,result)
    jitted_res = jit(hermitian)(x)
    assert np.allclose(jitted_res,jax_res)

def test_batched_hermitian():
    x_batched_resh = x_batched.reshape(5 , x_batched.shape[0]//5, *x_batched.shape[1:])
    vmapped = jax.vmap(hermitian)(x_batched_resh)
    vmapped_original = jax.vmap(origin_func)(x_batched_resh)
    assert np.allclose(vmapped,vmapped_original)

def test_gradients():

    grad_func = jacfwd(hermitian,holomorphic=True)(x)
    grad_origin = jacfwd(origin_func,holomorphic=True)(x)
    assert np.allclose(grad_func,grad_origin)

def test_rev_gradients():

    rev_grad_func = jacrev(hermitian,holomorphic=True)(x)
    rev_grad_origin = jacrev(origin_func,holomorphic=True)(x)
    assert np.allclose(rev_grad_func,rev_grad_origin)