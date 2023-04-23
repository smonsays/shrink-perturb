# Shrink-perturb 

[Optax](https://github.com/deepmind/optax) implementation of shrink and perturb ([Ash & Adams, 2020](https://arxiv.org/abs/1910.08475)).

## Example usage

```python
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from shrink_perturb import shrink_perturb

@hk.without_apply_rng
@hk.transform
def mlp(x):
    return hk.nets.MLP([10, 1])(x)


placeholder_input = jnp.empty((16, 16))
optimizer = optax.chain(
    optax.sgd(learning_rate=0.01),
    # Simply chain `shrink_and_perturb` after the optimizer
    # passing model init_fn closed over input
    shrink_perturb(
        param_init_fn=lambda k: mlp.init(k, placeholder_input),
        shrink=0.9,
        perturb=0.001,
    ),
)

params = mlp.init(jax.random.PRNGKey(0), placeholder_input)
optim_state = optimizer.init(params)
grads = jax.grad(lambda p, x: jnp.sum(mlp.apply(p, x)))(
    params, jnp.ones((16, 16))
)
# Need to pass params to optimizer.update()
params_update = optimizer.update(grads, optim_state, params)
```
