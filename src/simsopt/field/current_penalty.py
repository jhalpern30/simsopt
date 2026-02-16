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
    Penalize currents above a threshold using a one-sided quadratic penalty:

        J = sum_i max(0, I_i - I_threshold)^2
    """

    def __init__(self, currents, current_threshold):
        super().__init__(depends_on=currents)
        self.currents = list(currents)
        self.current_threshold = float(current_threshold)

        def _penalty(currents_vec):
            abs_currents = jnp.abs(currents_vec)
            excess = jnp.maximum(0.0, abs_currents - self.current_threshold)
            return jnp.sum(excess ** 2)

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
        self._dJ = Derivative(
            {c: np.atleast_1d(dJ_val[i]) for i, c in enumerate(self.currents)}
        )