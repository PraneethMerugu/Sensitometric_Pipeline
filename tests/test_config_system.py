import sys
import os
import jax.numpy as jnp
import jax
import pytest
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from core.config import load_config, FilmConfig
from pipeline import build_pipeline, run_one_off, FilmPipeline

def test_config_loading():
    config_path = "data/configs/default_vision3.json"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    
    config = load_config(config_path)
    assert isinstance(config, FilmConfig)
    assert config.name == "Kodak Vision3 250D (Default)"
    assert config.optical.scatter_gamma == 0.65
    # Check path resolution (should be absolute now, or at least relative to base_dir correctly)
    # The loader makes paths relative to the config file if they weren't absolute.
    # config file is in data/config/, paths starts with ../
    # so they should resolve to data/
    
    # We can check if the file exists using the resolved path
    assert os.path.exists(config.paths.cyan_density)

def test_pipeline_build():
    config_path = "data/configs/default_vision3.json"
    pipeline = build_pipeline(config_path)
    assert isinstance(pipeline, FilmPipeline)
    assert pipeline.optical.config.scatter_gamma == 0.65

def test_run_one_off():
    config_path = "data/configs/default_vision3.json"
    dummy_img = jnp.zeros((64, 64, 3))
    
    print("Running one-off...")
    output = run_one_off(config_path, dummy_img)
    
    assert output.shape == (64, 64, 3)
    assert not jnp.isnan(output).any()

if __name__ == "__main__":
    # key = jax.random.PRNGKey(0)
    test_config_loading()
    test_pipeline_build()
    test_run_one_off()
    print("All tests passed!")
