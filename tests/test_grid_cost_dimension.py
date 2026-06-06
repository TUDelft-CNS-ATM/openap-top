"""Tests for grid-cost interpolant dimensionality handling."""

import inspect

import pytest

import numpy as np
import opentop as top
from opentop import _objectives


def _identity_proj(x, y, inverse=False, symbolic=False):
    return x, y


def _interp_4d():
    return top.tools.construct_interpolant(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1000.0]),
        np.ones(16),
        timestamp=np.array([0.0, 10.0]),
        shape="linear",
    )


def test_grid_cost_auto_detects_4d_interpolant():
    x = np.array([0.5, 0.5, 500.0, 10_000.0, 5.0])
    u = np.array([0.7, 0.0, 0.0])

    cost = _objectives.obj_grid_cost(
        x,
        u,
        2.0,
        proj=_identity_proj,
        interpolant=_interp_4d(),
        time_dependent=True,
        symbolic=False,
    )

    assert cost == pytest.approx(2.0)


@pytest.mark.parametrize(
    "phase",
    [top.CompleteFlight, top.Cruise, top.Climb, top.Descent],
)
def test_phase_trajectory_n_dim_defaults_to_auto_detection(phase):
    param = inspect.signature(phase.trajectory).parameters["n_dim"]

    assert param.default is None
