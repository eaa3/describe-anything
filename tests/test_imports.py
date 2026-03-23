"""Tests for package imports and basic structure."""

import importlib
import subprocess
import sys


def test_dam_package_imports():
    """Core dam package and its public API should be importable."""
    import dam
    assert hasattr(dam, "DescribeAnythingModel")
    assert hasattr(dam, "disable_torch_init")


def test_dam_model_subpackage():
    """Model subpackage and key constants should be accessible."""
    from dam.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    assert isinstance(DEFAULT_IMAGE_TOKEN, str)
    assert isinstance(IMAGE_TOKEN_INDEX, int)


def test_dam_conversation():
    """Conversation module should load with separator styles."""
    from dam.model.conversation import SeparatorStyle
    assert hasattr(SeparatorStyle, "SINGLE")
    assert hasattr(SeparatorStyle, "TWO")


def test_llava_arch_no_init_weights_shim():
    """no_init_weights (or its shim) should work as a context manager."""
    from dam.model.llava_arch import no_init_weights
    import torch.nn as nn

    # Verify it works as a context manager without errors
    with no_init_weights(_enable=True):
        linear = nn.Linear(4, 2)
    assert linear.weight.shape == (2, 4)

    # Verify _enable=False is a no-op pass-through
    with no_init_weights(_enable=False):
        linear2 = nn.Linear(4, 2)
    assert linear2.weight.shape == (2, 4)


def test_dam_describe_anything_model_class():
    """DescribeAnythingModel class should exist with expected methods."""
    from dam import DescribeAnythingModel
    assert callable(DescribeAnythingModel)
    assert hasattr(DescribeAnythingModel, "get_description")
    assert hasattr(DescribeAnythingModel, "get_image_tensor")
    assert hasattr(DescribeAnythingModel, "crop_image")


def test_demo_video_main_exists():
    """demo_video module should have a callable main() for the entry point."""
    # Import just the module without executing main()
    spec = importlib.util.spec_from_file_location(
        "demo_video",
        "demo_video.py",
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    # We only need to check the source has 'def main', not actually run it
    # (running it would try to load models)
    import ast
    with open("demo_video.py") as f:
        tree = ast.parse(f.read())
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "main" in func_names, "demo_video.py must define a main() function"


def test_entry_point_help():
    """The dam-video-demo entry point should respond to --help."""
    result = subprocess.run(
        [sys.executable, "demo_video.py", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "Describe Anything Video Demo" in result.stdout


def test_pyproject_metadata():
    """pyproject.toml should have correct build backend and metadata."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    assert data["build-system"]["build-backend"] == "hatchling.build"
    assert data["project"]["name"] == "dam"
    assert ">=3.10" in data["project"]["requires-python"]
    assert "sam2" in data["project"]["optional-dependencies"]
    assert "sam" in data["project"]["optional-dependencies"]
    assert data["tool"]["hatch"]["metadata"]["allow-direct-references"] is True
