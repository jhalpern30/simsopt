import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "CurrentPenalty requires JAX. Install with the appropriate simsopt extras."
    ) from exc

__all__ = ['CurrentPenalty']

class CurrentPenalty(Optimizable):
    """
    Penalize coil currents using a global p-norm of the absolute current vector:

        J = (sum_i |I_i|^p)^(1/p)

    This objective is smooth for p > 1 and serves as a differentiable proxy for
    the maximum current (as p increases, the p-norm approaches max_i |I_i|).
    """

    def __init__(self, currents, p=2.0):
        super().__init__(depends_on=currents)
        self.currents = list(currents)
        self.p = float(p)
        if not np.isfinite(self.p) or self.p <= 1.0:
            raise ValueError("CurrentPenalty requires p > 1 for a well-behaved gradient.")

        def _penalty(currents_vec):
            abs_currents = jnp.abs(currents_vec)
            return jnp.power(jnp.sum(jnp.power(abs_currents, self.p)), 1.0 / self.p)

        self._jax_penalty = _penalty
        self._jax_grad = jax.grad(_penalty)

        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        currents_vec = jnp.asarray(
            [c.get_value() for c in self.currents], dtype=jnp.float64
        )
        J_val = self._jax_penalty(currents_vec)
        dJ_val = np.asarray(self._jax_grad(currents_vec), dtype=float)

        self._J = float(J_val)
        deriv = Derivative({})
        for i, c in enumerate(self.currents):
            deriv += c.vjp(np.atleast_1d(dJ_val[i]))
        self._dJ = deriv