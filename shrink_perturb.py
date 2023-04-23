from typing import Callable, NamedTuple

import chex
import jax
from optax import GradientTransformation


class ShrinkPerturbState(NamedTuple):
    rng_key: chex.PRNGKey


def shrink_perturb(
    param_init_fn: Callable, shrink: float = 0.8, perturb: float = 0.01, seed: int = 0
) -> GradientTransformation:
    """Shrink and perturb.
    References:
        [Ash & Adams, 2020](https://arxiv.org/abs/1910.08475)
    Args:
        param_init_fn: Function to initialize params only taking rng as input
        shrink: Amount of shrinking
        perturb: Perturbation amount
        seed: Seed for random number generation.
    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return ShrinkPerturbState(rng_key=jax.random.PRNGKey(seed))

    def update_fn(updates, state, params):  # pylint: disable=missing-docstring
        if params is None:
            raise ValueError(
                "You are using a transformation that requires the current value of "
                "parameters, but you are not passing `params` when calling `update`."
            )
        new_rng, rng_init = jax.random.split(state.rng_key, num=2)
        noise = param_init_fn(rng_init)
        updates = jax.tree_util.tree_map(
            lambda g, n, p: g + perturb * n + (shrink - 1.0) * p, updates, noise, params
        )
        return updates, ShrinkPerturbState(rng_key=new_rng)

    return GradientTransformation(init_fn, update_fn)
