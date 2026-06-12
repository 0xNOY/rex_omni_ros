"""Tests for the GPU-less parts of the engine (auto VRAM sizing)."""

from pathlib import Path

import pytest

from rex_omni_ros.core.engine import (
    GIB,
    EngineConfig,
    _checkpoint_weight_nbytes,
    auto_gpu_memory_utilization,
)

WEIGHT_NBYTES = int(3.5 * GIB)  # ≈ the AWQ checkpoint
TOTAL_VRAM = 24 * GIB


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    # Sparse files: st_size is what the estimator reads, no disk is used.
    with (tmp_path / "model.safetensors").open("wb") as file:
        file.truncate(WEIGHT_NBYTES)
    return tmp_path


def config_for(checkpoint: Path, **overrides: object) -> EngineConfig:
    return EngineConfig(
        model_path=str(checkpoint),
        gpu_memory_utilization=0.0,
        **overrides,  # type: ignore[arg-type]
    )


def test_weight_nbytes_sums_weight_files(checkpoint: Path) -> None:
    with (checkpoint / "extra.bin").open("wb") as file:
        file.truncate(GIB)
    (checkpoint / "config.json").write_text("{}")  # not a weight file

    assert _checkpoint_weight_nbytes(str(checkpoint)) == WEIGHT_NBYTES + GIB


def test_weight_nbytes_rejects_checkpoint_without_weights(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no weight files"):
        _checkpoint_weight_nbytes(str(tmp_path))


def test_auto_utilization_is_a_sane_fraction(checkpoint: Path) -> None:
    utilization = auto_gpu_memory_utilization(config_for(checkpoint), TOTAL_VRAM)

    # Weights alone are ~3.5 GiB; everything together stays well under 50%
    # of a 24 GiB GPU (measured minimum for the default config is ~6 GiB).
    assert WEIGHT_NBYTES / TOTAL_VRAM < utilization < 0.5


def test_auto_utilization_grows_with_context_and_image_budget(
    checkpoint: Path,
) -> None:
    base = auto_gpu_memory_utilization(config_for(checkpoint), TOTAL_VRAM)
    more_context = auto_gpu_memory_utilization(
        config_for(checkpoint, max_model_len=8192), TOTAL_VRAM
    )
    more_pixels = auto_gpu_memory_utilization(
        config_for(checkpoint, max_pixels=4 * 2007040), TOTAL_VRAM
    )

    assert more_context > base
    assert more_pixels > base


def test_auto_utilization_drops_cuda_graphs_when_eager(checkpoint: Path) -> None:
    default = auto_gpu_memory_utilization(config_for(checkpoint), TOTAL_VRAM)
    eager = auto_gpu_memory_utilization(
        config_for(checkpoint, enforce_eager=True), TOTAL_VRAM
    )

    assert eager < default


def test_auto_utilization_rejects_too_small_gpu(checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="GiB VRAM"):
        auto_gpu_memory_utilization(config_for(checkpoint), 4 * GIB)
