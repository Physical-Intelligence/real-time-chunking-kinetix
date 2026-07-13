"""RTC carry-buffer regression tests for real-time-chunking-kinetix#9."""

import jax.numpy as jnp
import numpy as np


def roll_action_chunk(
    prev_chunk: jnp.ndarray,
    corrected_chunk: jnp.ndarray,
    *,
    execute_horizon: int,
    chunk_size: int,
) -> jnp.ndarray:
    """Match eval_flow.execute_chunk carry update."""
    return jnp.concatenate([prev_chunk, corrected_chunk], axis=1)[
        :, execute_horizon : execute_horizon + chunk_size
    ]


def test_roll_preserves_inference_delay_overlap():
    # chunk_size=8, execute_horizon=4, inference_delay=2
    prev = jnp.arange(8 * 3, dtype=jnp.float32).reshape(1, 8, 3)
    corrected = (jnp.arange(8 * 3, dtype=jnp.float32) + 100).reshape(1, 8, 3)
    rolled = roll_action_chunk(prev, corrected, execute_horizon=4, chunk_size=8)

    expected = jnp.concatenate([prev, corrected], axis=1)[:, 4:12]
    assert np.allclose(np.asarray(rolled), np.asarray(expected))

    # Prefix continuity: first slot of rolled chunk should match prev[4], not corrected[4].
    assert float(rolled[0, 0, 0]) == float(prev[0, 4, 0])


def test_old_carry_logic_breaks_prefix_continuity():
    prev = jnp.arange(8 * 3, dtype=jnp.float32).reshape(1, 8, 3)
    corrected = (jnp.arange(8 * 3, dtype=jnp.float32) + 100).reshape(1, 8, 3)
    broken = jnp.concatenate(
        [
            corrected[:, 4:],
            jnp.zeros((1, 4, 3)),
        ],
        axis=1,
    )
    rolled = roll_action_chunk(prev, corrected, execute_horizon=4, chunk_size=8)
    assert not np.allclose(np.asarray(broken[:, :4]), np.asarray(rolled[:, :4]))
