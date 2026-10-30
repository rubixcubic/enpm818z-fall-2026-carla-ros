"""GP3: GNSS to the map frame, and the outage window.

    cd ads_pipeline && python3 -m pytest test/test_gp3_geo_outage.py -v
"""
import pytest

from ads_pipeline.geo import MapProjection
from ads_pipeline.gnss_outage import in_outage


def test_origin_maps_to_zero():
    p = MapProjection(0.0, 0.0)
    assert p.to_map(0.0, 0.0) == pytest.approx((0.0, 0.0))


def test_carla_latitude_grows_toward_minus_y_and_longitude_toward_plus_x():
    """Measured on Town01 (team_drives/check_frames.py): a fix north of the
    origin is at the map frame's -y, because the map frame flips CARLA's y."""
    p = MapProjection(0.0, 0.0)
    x, y = p.to_map(1e-5, 1e-5)
    assert y < -1.0 and x > 1.0


def test_scale_is_carlas():
    """CARLA's transform_to_geolocation on Town01: 100 m east of the origin
    is longitude 0.0008986747539842821 degrees."""
    p = MapProjection(0.0, 0.0)
    assert p.to_map(0.0, 0.0008986747539842821)[0] == pytest.approx(100.0, abs=1e-3)
    assert p.to_map(0.00089867475402113, 0.0)[1] == pytest.approx(-100.0, abs=1e-3)


def test_outage_window_is_relative_to_first_fix():
    assert not in_outage(100.0, 100.0, 30.0, 50.0)
    assert in_outage(135.0, 100.0, 30.0, 50.0)
    assert not in_outage(150.0, 100.0, 30.0, 50.0)   # end is open
