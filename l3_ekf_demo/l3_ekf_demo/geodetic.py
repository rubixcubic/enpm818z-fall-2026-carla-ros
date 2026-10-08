#!/usr/bin/env python3
"""Latitude and longitude to local meters.

Slide: 'CARLA Hands-On', Task 1, the line 'CARLA's GNSS reports latitude and
longitude, so you must convert to local meters first.'

This is the trap that costs people an evening. A NavSatFix is degrees. The
filter's state is meters. Subtracting two latitudes and calling the answer a
distance gives you numbers that are wrong by a factor of about 111,000, and
nothing raises an error: the filter runs, converges, and is nonsense.

Flat-earth (equirectangular) projection about a fixed anchor. Good to well
under a centimeter over a CARLA town, which is a few hundred meters across.
Do NOT reuse this over tens of kilometers.
"""

import math

WGS84_A = 6378137.0          # semi-major axis, meters


class LocalENU:
    """Converts WGS84 lat/lon to local east/north meters about an anchor.

    The anchor is the origin of the road frame from the Terminology section:
    one point, fixed once, and every coordinate afterwards is relative to it.
    Move the anchor and every number changes while the car has not moved.
    """

    def __init__(self, lat0_deg: float = 0.0, lon0_deg: float = 0.0):
        self._lat0 = math.radians(lat0_deg)
        self._lon0 = math.radians(lon0_deg)
        self._latched = False
        # meters per radian, evaluated once at the anchor latitude
        self._m_per_rad_north = WGS84_A
        self._m_per_rad_east = WGS84_A * math.cos(self._lat0)

    @property
    def latched(self) -> bool:
        return self._latched

    def latch(self, lat_deg: float, lon_deg: float) -> None:
        """Adopt this fix as the local origin."""
        self._lat0 = math.radians(lat_deg)
        self._lon0 = math.radians(lon_deg)
        self._m_per_rad_north = WGS84_A
        self._m_per_rad_east = WGS84_A * math.cos(self._lat0)
        self._latched = True

    def to_enu(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        """Return (east, north) in meters relative to the anchor."""
        dlat = math.radians(lat_deg) - self._lat0
        dlon = math.radians(lon_deg) - self._lon0
        return dlon * self._m_per_rad_east, dlat * self._m_per_rad_north
