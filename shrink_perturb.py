"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Callable, NamedTuple

import chex
import jax
from optax import GradientTransformation


class ShrinkPerturbState(NamedTuple):
    count: chex.Array
    rng_key: chex.PRNGKey


def shrink_and_perturb(
    param_init_fn: Callable,
    shrink: float = 0.8,
    perturb: float = 0.01,
    every_n: int = 1,
    seed: int = 0,
) -> GradientTransformation:
    """Shrink and perturb.
    References:
        [Ash & Adams, 2020](https://arxiv.org/abs/1910.08475)
    Args:
        param_init_fn: Function to initialize params only taking rng as input
        shrink: Amount of shrinking
        perturb: Perturbation amount
        every_n: only apply every nth update
        seed: Seed for random number generation.
    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return ShrinkPerturbState(count=jnp.zeros([], jnp.int32), rng_key=jax.random.PRNGKey(seed))

    def update_fn(updates, state, params):  # pylint: disable=missing-docstring
        if params is None:
            raise ValueError(
                'You are using a transformation that requires the current value of '
                'parameters, but you are not passing `params` when calling `update`.'
            )
        new_rng, rng_init = jax.random.split(state.rng_key, num=2)
        noise = param_init_fn(rng_init)
        mask = (state.count % every_n == 0) and (state.count > 0)
        updates = jax.tree_util.tree_map(
            lambda g, n, p: g + mask * (perturb * n + (shrink - 1.0) * p), updates, noise, params
        )
        return updates, ShrinkPerturbState(count=(state.count + 1) % every_n, rng_key=new_rng)

    return GradientTransformation(init_fn, update_fn)
