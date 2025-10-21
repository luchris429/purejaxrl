import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import jax
import jax.numpy as jnp
from jax import random


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def rng_key():
    """Provide a JAX random key for tests."""
    return random.PRNGKey(42)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing."""
    return {
        "learning_rate": 3e-4,
        "num_envs": 4,
        "num_steps": 128,
        "total_timesteps": 1_000_000,
        "update_epochs": 4,
        "num_minibatches": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "seed": 42,
    }


@pytest.fixture
def sample_observation():
    """Provide a sample observation for testing."""
    return jnp.array([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
def sample_action():
    """Provide a sample action for testing."""
    return jnp.array(1)


@pytest.fixture
def sample_batch():
    """Provide a sample batch of data for testing."""
    batch_size = 32
    obs_dim = 4
    return {
        "obs": jnp.ones((batch_size, obs_dim)),
        "actions": jnp.zeros(batch_size, dtype=jnp.int32),
        "rewards": jnp.ones(batch_size),
        "dones": jnp.zeros(batch_size, dtype=jnp.bool_),
        "values": jnp.ones(batch_size),
        "log_probs": jnp.zeros(batch_size),
    }


@pytest.fixture(autouse=True)
def setup_jax():
    """Configure JAX for testing."""
    # Disable JIT compilation for easier debugging in tests
    os.environ["JAX_DISABLE_JIT"] = "0"  # Can be set to "1" for debugging
    
    # Use CPU by default for tests
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    
    yield
    
    # Cleanup after tests
    if "JAX_DISABLE_JIT" in os.environ:
        del os.environ["JAX_DISABLE_JIT"]
    if "JAX_PLATFORM_NAME" in os.environ:
        del os.environ["JAX_PLATFORM_NAME"]


@pytest.fixture
def mock_env_params():
    """Provide mock environment parameters."""
    return {
        "max_steps": 200,
        "obs_shape": (4,),
        "action_shape": (),
        "num_actions": 2,
    }


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages during tests."""
    caplog.set_level("DEBUG")
    return caplog


@pytest.fixture
def wandb_mock(mocker):
    """Mock Weights & Biases for testing."""
    mock_wandb = mocker.patch("wandb.init")
    mock_wandb.return_value = mocker.MagicMock()
    mocker.patch("wandb.log")
    mocker.patch("wandb.finish")
    return mock_wandb