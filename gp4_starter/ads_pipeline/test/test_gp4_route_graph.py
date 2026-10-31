"""GP4: the waypoint graph (given) that the planner searches.

    cd ads_pipeline && python3 -m pytest test/test_gp4_route_graph.py -v
"""
from pathlib import Path

import pytest

from ads_pipeline.route_graph import RouteGraph

MAPS = Path(__file__).resolve().parents[1] / 'ads_pipeline' / 'maps'


@pytest.fixture(scope='module')
def town01():
    return RouteGraph.from_xodr(MAPS / 'Town01.xodr')


def test_town01_node_count(town01):
    # 3107: the number of nodes in the course's Town01 graph (2 m sampling)
    assert len(town01.xy) == 3107


def test_town01_has_no_lane_change(town01):
    assert not any(lc for _, _, _, lc in town01.edges)


def test_town03_has_lane_changes():
    g = RouteGraph.from_xodr(MAPS / 'Town03.xodr')
    assert sum(1 for _, _, _, lc in g.edges if lc) > 0


def test_nearest_node_of_a_node_is_itself(town01):
    assert town01.nearest(*town01.xy[100]) == 100


def test_cost_is_the_edge_length(town01):
    a, b, length, _ = town01.edges[0]
    assert town01.cost(a, b) == pytest.approx(length)


def test_closing_edges_removes_only_nearby_ones():
    g = RouteGraph.from_xodr(MAPS / 'Town01.xodr')
    n0 = sum(len(g.neighbors(i)) for i in range(len(g.xy)))
    removed = g.close_edges_near(*g.xy[500], 3.0)
    n1 = sum(len(g.neighbors(i)) for i in range(len(g.xy)))
    assert removed > 0 and n0 - n1 == removed


def test_close_node_cuts_every_edge_into_and_out_of_it():
    g = RouteGraph.from_xodr(MAPS / 'Town01.xodr')
    g.close_node(1000)
    assert g.neighbors(1000) == []
    assert all(j != 1000 for i in range(len(g.xy)) for j, _ in g.neighbors(i))


def test_junction_ids_mark_junction_nodes(town01):
    j = town01.junction_id
    assert len(j) == len(town01.xy)
    assert (j >= 0).sum() > 0 and (j < 0).sum() > 0       # some nodes in junctions, most not


def test_closed_edges_can_be_reopened():
    g = RouteGraph.from_xodr(MAPS / 'Town01.xodr')
    n0 = sum(len(g.neighbors(i)) for i in range(len(g.xy)))
    closed = []
    g.close_edges_near(*g.xy[500], 3.0, record=closed)
    g.reopen(closed)
    assert sum(len(g.neighbors(i)) for i in range(len(g.xy))) == n0
