from deprecated import deprecated

from jax import grad
import jax.numpy as jnp

from ..geo.jit import jit
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
import simsoptpp as sopp

__all__ = ['CurrentPenalty', 'SquaredCurrent']

@jit
def current_penalty_pure(I, p, threshold):
    return jnp.maximum(abs(I) - threshold, 0)**p

class CurrentPenalty(Optimizable):
    """
    A :obj:`CurrentPenalty` can be used to penalize
    large currents in coils.
    """
    def __init__(self, current, p, threshold=0):
        self.current = current
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[current])
        self.J_jax = jit(lambda c, p, t: current_penalty_pure(c, p, t))
        self.this_grad = jit(lambda c, p, t: grad(self.J_jax, argnums=0)(c, p, t))

    def J(self):
        return self.J_jax(self.current.get_value(), self.p, self.threshold)

    @derivative_dec
    def dJ(self):
        grad0 = self.this_grad(self.current.get_value(), self.p, self.threshold)
        return self.current.vjp(grad0)
    
# def current_penalty_pure(I, threshold):
#     return jnp.maximum(abs(I) - threshold, 0)**2

# class CurrentPenalty(Optimizable):
#     """
#     A :obj:`CurrentPenalty` can be used to penalize
#     large currents in coils.
#     """
#     def __init__(self, current, threshold=0):
#         self.current = current
#         self.threshold = threshold
#         super().__init__(depends_on=[current])
#         self.J_jax = lambda I: current_penalty_pure(I, self.threshold)
#         self.this_grad = lambda I: grad(self.J_jax, argnums=0)(I)
#     def J(self):
#         return self.J_jax(self.current.get_value())
#     @derivative_dec
#     def dJ(self):
#         grad0 = self.this_grad(self.current.get_value())
#         return self.current.vjp(grad0)
    
def squared_current_pure(I):
    return I**2

class SquaredCurrent(Optimizable):
    """
    A :obj:`CurrentPenalty` can be used to penalize
    total squared current in the coils
    """
    def __init__(self, current):
        self.current = current
        super().__init__(depends_on=[current])
        self.J_jax = lambda I: squared_current_pure(I)
        self.this_grad = lambda I: grad(self.J_jax, argnums=0)(I)
    def J(self):
        return self.J_jax(self.current.get_value())
    @derivative_dec
    def dJ(self):
        grad0 = self.this_grad(self.current.get_value())
        return self.current.vjp(grad0)