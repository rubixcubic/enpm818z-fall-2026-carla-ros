"""The demo plans with the slides' numbers (no CARLA server needed)."""
from pathlib import Path

from l9_planning_demo import planner as pl

DATA = Path(__file__).resolve().parents[1] / "data" / "l9_scene.npz"


def test_cheapest_path_is_the_slides():
    ref = pl.load_scene(DATA)
    car = pl.parked_car(ref)
    out = pl.plan(ref, pl.SCENE["av_s"], (0.0, 0.0, 0.0), [car])
    best, sc = out[0]
    assert (best["d_end"], best["maneuver"]) == (3.5, 40.0)
    assert round(sc["total"], 2) == 5.07
    assert len(out) == 31
    assert sum(1 for _, s in out if s["dropped"]) == 22


def test_larger_clearance_weight_moves_the_path_left():
    ref = pl.load_scene(DATA)
    car = pl.parked_car(ref)
    w = dict(pl.WEIGHTS, clearance=10.0)
    best, sc = pl.plan(ref, pl.SCENE["av_s"], (0.0, 0.0, 0.0), [car], weights=w)[0]
    assert (best["d_end"], best["maneuver"]) == (4.0, 40.0)
    assert round(sc["total"], 2) == 5.86
