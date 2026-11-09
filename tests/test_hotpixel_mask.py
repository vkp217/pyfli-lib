"""
pyfli.io.hotpixel: mask generation (fraction and MAD rules), image round trip, and
order-independent neighbour-median removal for images and cubes.
"""

import numpy as np
import pytest

from pyfli.io import (
    DataIOUtils,
    generate_hotpixel_mask,
    hotpixel_counts,
    load_hotpixel_mask,
    remove_hotpixels,
    save_hotpixel_mask,
)
from pyfli.io.data_ops_static import StaticDataOps


def _dark_cube(seed=0, shape=(40, 50, 8), n_hot=60):
    rng = np.random.default_rng(seed)
    cube = rng.poisson(3.0, shape).astype(float)
    hot = np.zeros(shape[:2], bool)
    hot.flat[rng.choice(hot.size, n_hot, replace=False)] = True
    cube[hot] += rng.uniform(50, 200, (n_hot, shape[2]))
    return cube, hot


def test_counts_shapes():
    """Images pass through, cubes and stacks are summed to (H, W)."""
    cube, _ = _dark_cube()
    assert hotpixel_counts(cube).shape == (40, 50)
    np.testing.assert_allclose(
        hotpixel_counts(cube[None].repeat(2, 0)), 2 * cube.sum(-1)
    )
    with pytest.raises(ValueError):
        hotpixel_counts(np.zeros(5))


def test_fraction_rule_exact_count_and_finds_hot_pixels():
    """The fraction rule flags exactly round(f * N) pixels, the hottest ones."""
    cube, hot = _dark_cube()
    mask = generate_hotpixel_mask(cube, fraction=60 / 2000)
    assert mask.sum() == 60 and np.array_equal(mask, hot)
    assert generate_hotpixel_mask(cube, fraction=0.05).sum() == 100
    assert generate_hotpixel_mask(cube, fraction=0.0).sum() == 0


def test_mad_rule_matches_dataioutils():
    """threshold_sigma uses the same MAD rule as DataIOUtils.hot_pixels."""
    cube, hot = _dark_cube()
    mask = generate_hotpixel_mask(cube, threshold_sigma=5.0)
    np.testing.assert_array_equal(
        mask, DataIOUtils.hot_pixels(cube, threshold_sigma=5.0)[0]
    )
    assert np.array_equal(mask, hot)


def test_save_load_round_trip(tmp_path):
    """A saved mask loads back identically, transposed when the data shape asks for it."""
    _, hot = _dark_cube()
    path = str(tmp_path / "mask.png")
    generate_hotpixel_mask(_dark_cube()[0], fraction=0.03, save_path=path)
    save_hotpixel_mask(hot, path)
    np.testing.assert_array_equal(load_hotpixel_mask(path), hot)
    np.testing.assert_array_equal(load_hotpixel_mask(path, shape=(50, 40)), hot.T)
    with pytest.raises(ValueError):
        load_hotpixel_mask(path, shape=(10, 10))


def test_removal_isolated_matches_existing_correction():
    """Isolated hot pixels get the 3x3 neighbour median, as StaticDataOps.hotpixel_correct."""
    rng = np.random.default_rng(1)
    cube = rng.poisson(20.0, (30, 30, 4)).astype(float)
    mask = np.zeros((30, 30), bool)
    mask[[5, 12, 20, 0, 29], [7, 15, 3, 0, 29]] = True
    cube[mask] = 1e4
    out = remove_hotpixels(cube, mask)
    np.testing.assert_allclose(
        out, StaticDataOps.hotpixel_correct(cube, mask), rtol=1e-6
    )
    np.testing.assert_array_equal(remove_hotpixels(cube[..., 2], mask), out[..., 2])
    assert np.array_equal(out[~mask], cube[~mask].astype(np.float32))


def test_removal_ignores_hot_neighbours_and_widens():
    """Clusters use only non-hot neighbours; a fully hot 3x3 widens to 5x5."""
    img = np.full((9, 9), 10.0)
    mask = np.zeros((9, 9), bool)
    mask[3:6, 3:6] = True
    img[mask] = 1e4
    out = remove_hotpixels(img, mask)
    np.testing.assert_array_equal(out, np.full((9, 9), 10.0, dtype=np.float32))
    with pytest.raises(ValueError):
        remove_hotpixels(img, mask[:5])
