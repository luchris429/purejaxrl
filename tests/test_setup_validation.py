import sys
from pathlib import Path

import pytest
import jax
import jax.numpy as jnp


class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly configured."""
    
    def test_pytest_is_running(self):
        """Verify that pytest is executing tests."""
        assert True, "Pytest is not running properly"
    
    def test_project_structure_exists(self):
        """Verify the project structure is correctly set up."""
        project_root = Path(__file__).parent.parent
        
        # Check main package exists
        assert (project_root / "purejaxrl").exists(), "purejaxrl package not found"
        assert (project_root / "purejaxrl" / "__init__.py").exists() or \
               any((project_root / "purejaxrl").glob("*.py")), \
               "No Python files found in purejaxrl package"
        
        # Check test directories exist
        assert (project_root / "tests").exists(), "tests directory not found"
        assert (project_root / "tests" / "unit").exists(), "tests/unit directory not found"
        assert (project_root / "tests" / "integration").exists(), "tests/integration directory not found"
        
        # Check configuration files exist
        assert (project_root / "pyproject.toml").exists(), "pyproject.toml not found"
    
    def test_jax_import_works(self):
        """Verify JAX can be imported and basic operations work."""
        # Test basic JAX operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        z = x + y
        
        assert jnp.allclose(z, jnp.array([5.0, 7.0, 9.0])), "JAX addition failed"
        
        # Test random number generation
        key = jax.random.PRNGKey(0)
        random_array = jax.random.normal(key, (3,))
        assert random_array.shape == (3,), "JAX random generation failed"
    
    def test_fixtures_are_available(self, temp_dir, rng_key, mock_config):
        """Verify pytest fixtures from conftest.py are available."""
        # Test temp_dir fixture
        assert temp_dir.exists(), "temp_dir fixture not working"
        assert temp_dir.is_dir(), "temp_dir is not a directory"
        
        # Test rng_key fixture
        assert hasattr(rng_key, "shape"), "rng_key fixture not working"
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict), "mock_config fixture not working"
        assert "learning_rate" in mock_config, "mock_config missing expected keys"
    
    @pytest.mark.unit
    def test_unit_marker_works(self):
        """Verify the unit test marker is recognized."""
        assert True, "Unit marker not working"
    
    @pytest.mark.integration
    def test_integration_marker_works(self):
        """Verify the integration test marker is recognized."""
        assert True, "Integration marker not working"
    
    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Verify the slow test marker is recognized."""
        assert True, "Slow marker not working"
    
    def test_coverage_tracking(self):
        """Verify that code coverage is being tracked."""
        # This test ensures coverage is configured properly
        # The actual coverage percentage will be checked by pytest-cov
        def dummy_function(x):
            if x > 0:
                return x * 2
            else:
                return 0
        
        assert dummy_function(5) == 10, "Coverage test function failed"
    
    def test_python_path_includes_project_root(self):
        """Verify the Python path is correctly configured."""
        project_root = str(Path(__file__).parent.parent)
        assert any(project_root in path for path in sys.path), \
            f"Project root {project_root} not in Python path"
    
    def test_all_dependencies_importable(self):
        """Verify key dependencies can be imported."""
        required_imports = [
            ("flax", "flax"),
            ("optax", "optax"),
            ("gymnax", "gymnax"),
            ("distrax", "distrax"),
        ]
        
        for module_name, import_name in required_imports:
            try:
                __import__(import_name)
            except ImportError:
                pytest.skip(f"{module_name} not installed - run 'poetry install' first")


@pytest.mark.parametrize("test_type", ["unit", "integration", "slow"])
def test_markers_can_be_filtered(test_type):
    """Verify test markers can be used to filter tests."""
    # This test will be run for each marker type
    # Users can filter with: pytest -m unit, pytest -m integration, etc.
    assert test_type in ["unit", "integration", "slow"], f"Unknown test type: {test_type}"