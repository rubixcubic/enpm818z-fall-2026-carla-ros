"""GP4 Task 2 tests: A* on the waypoint graph, against CARLA's own planner.

The expected lengths were computed once with CARLA's GlobalRoutePlanner
(2 m sampling, distance along the route). On Town01, which has no lane
changes, both graphs are the same, so the lengths agree within 4 m (2 m of
snapping at each end). On Town03, CARLA's graph links a lane change only
once per road segment and costs it 0; ours allows one wherever the road
markings do and costs it its sideways distance. A legal route can then be
shorter than CARLA's, never longer than it by more than a lane width (4 m)
per lane change.

    cd ads_pipeline && python3 -m pytest test/test_gp4_planner.py -v
"""
import json
from pathlib import Path

import pytest

from ads_pipeline.planner_core import astar_waypoints
from ads_pipeline.route_graph import RouteGraph

ROOT = Path(__file__).resolve().parents[1]
ROUTES = json.loads((ROOT / 'test_data' / 'grp_routes.json').read_text())
GRAPHS = {}


def graph(town):
    if town not in GRAPHS:
        GRAPHS[town] = RouteGraph.from_xodr(ROOT / 'ads_pipeline' / 'maps' / f'{town}.xodr')
    return GRAPHS[town]


@pytest.mark.parametrize('r', ROUTES, ids=lambda r: f"{r['town']}-{r['grp_length_m']:.0f}m")
def test_route_matches_carlas_planner(r):
    g = graph(r['town'])
    path, _ = astar_waypoints(g, g.nearest(*r['start_xy']), g.nearest(*r['goal_xy']))
    length = sum(g.cost(a, b) for a, b in zip(path, path[1:]))
    if r['lane_changes'] == 0:
        assert abs(length - r['grp_length_m']) <= 4.0
    else:
        assert length <= r['grp_length_m'] + 4.0 * r['lane_changes'] + 4.0


def test_unreachable_goal_returns_none():
    g = RouteGraph.from_xodr(ROOT / 'ads_pipeline' / 'maps' / 'Town01.xodr')
    g.close_node(1000)
    path, _ = astar_waypoints(g, 0, 1000)
    assert path is None


def test_search_stops_at_max_nodes():
    g = graph('Town01')
    path, expanded = astar_waypoints(g, 0, len(g.xy) - 1, max_nodes=10)
    assert path is None and expanded <= 10


def test_path_starts_and_ends_where_asked():
    g = graph('Town01')
    r = ROUTES[0]
    s, t = g.nearest(*r['start_xy']), g.nearest(*r['goal_xy'])
    path, _ = astar_waypoints(g, s, t)
    assert path[0] == s and path[-1] == t
