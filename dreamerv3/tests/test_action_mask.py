import jax
import jax.numpy as jnp
import numpy as np

from embodied.jax.outs import MaskedCategorical


def test_masked_categorical_never_selects_invalid_actions():
  logits = jnp.array([[20.0, 2.0, 1.0], [1.0, 20.0, 3.0]])
  mask = jnp.array([[False, True, True], [True, False, False]])
  dist = MaskedCategorical(logits, mask, unimix=0.1)

  samples = dist.sample(jax.random.PRNGKey(0), (1000,))
  assert np.all(np.asarray(samples[:, 0]) != 0)
  assert np.all(np.asarray(samples[:, 1]) == 0)
  assert np.asarray(dist.pred()).tolist() == [1, 0]


def test_masked_categorical_unimix_and_gradients_stay_inside_valid_set():
  logits = jnp.array([10.0, 2.0, 1.0])
  mask = jnp.array([False, True, True])
  dist = MaskedCategorical(logits, mask, unimix=0.2)
  probs = jax.nn.softmax(dist.logits)

  assert float(probs[0]) == 0.0
  assert np.isclose(float(probs[1] + probs[2]), 1.0)

  grad = jax.grad(
      lambda value: MaskedCategorical(
          value, mask, unimix=0.2).logp(jnp.array(1)))(logits)
  assert np.isfinite(np.asarray(grad)).all()
  assert float(grad[0]) == 0.0
