"""Latitude and longitude to the map frame. Given.

The anchor is the latitude and longitude of the map's origin, from
ekf_config.yaml (gnss_origin_lat, gnss_origin_lon); Town01's is 0, 0.
Anchoring at the first fix instead would put the EKF in a different frame
from the ground truth.

Two facts about CARLA, measured on Town01 with transform_to_geolocation and
a recorded drive (instructor's check_frames.py):

- 100 m along an axis is 0.0008986747539842821 degrees, so one degree is
  100 / 0.0008986747539842821 = 111,275.5 m near the origin.
- Latitude grows toward the map frame's -y. The map frame is CARLA's with
  y flipped (GP1's TF), and CARLA's latitude grows with CARLA's +y.
"""
import math

M_PER_DEG = 100.0 / 0.0008986747539842821


class MapProjection:
    def __init__(self, lat0_deg, lon0_deg):
        self.lat0, self.lon0 = lat0_deg, lon0_deg
        self._m_east = M_PER_DEG * math.cos(math.radians(lat0_deg))

    def to_map(self, lat_deg, lon_deg):
        """(x, y) in meters in the map frame."""
        return ((lon_deg - self.lon0) * self._m_east,
                -(lat_deg - self.lat0) * M_PER_DEG)
