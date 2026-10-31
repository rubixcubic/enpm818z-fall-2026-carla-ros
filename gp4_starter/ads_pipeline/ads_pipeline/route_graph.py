"""The waypoint graph the GP4 planner searches. Given.

Built offline from a map file with carla.Map(name, xodr_text): no CARLA
server. The graph is sampled the way CARLA's GlobalRoutePlanner samples it:

- one topology segment is one lane from one junction boundary to the next
  (or one lane through a junction); its waypoints are RES meters apart;
- a segment's end and the next segment's start are the same point, so they
  are one node (locations rounded to 0.1 m);
- an edge from each waypoint to the next one along its lane, as long as the
  straight line between them;
- outside junctions, an edge to the neighbor lane where the map allows a
  lane change, as long as the sideways distance.

Nodes are in the map frame: x as CARLA's, y = -y_carla.
"""
from pathlib import Path

import carla
import numpy as np
from scipy.spatial import cKDTree

RES = 2.0


class RouteGraph:
    def __init__(self, xy, edges, junction_id=None):
        self.xy = np.asarray(xy, dtype=float)
        # junction_id[i]: the junction node i lies in, -1 outside junctions
        self.junction_id = (np.full(len(self.xy), -1) if junction_id is None
                            else np.asarray(junction_id))
        self.edges = list(edges)                      # (a, b, length, lane_change)
        self._out = [dict() for _ in range(len(self.xy))]
        for a, b, length, _ in self.edges:
            self._out[a][b] = length
        self._tree = cKDTree(self.xy)

    @classmethod
    def from_xodr(cls, path, resolution=RES):
        path = Path(path)
        m = carla.Map(path.stem, path.read_text())
        xy, keys, wps = [], {}, []

        def node(w, shared):
            loc = w.transform.location
            if shared:
                k = ('end', round(loc.x, 1), round(loc.y, 1))
            else:
                k = ('in', w.road_id, w.section_id, w.lane_id, round(w.s, 1))
            if k not in keys:
                keys[k] = len(xy)
                xy.append((loc.x, -loc.y))
                wps.append(w)
            return keys[k]

        edges, seen = [], set()
        for a, b in m.get_topology():
            end = b.transform.location
            chain = [a]
            if a.transform.location.distance(end) > resolution:
                w = a.next(resolution)[0]
                while w.transform.location.distance(end) > resolution:
                    chain.append(w)
                    nxt = w.next(resolution)
                    if not nxt:
                        break
                    w = nxt[0]
            chain.append(b)
            ids = ([node(chain[0], True)] + [node(w, False) for w in chain[1:-1]]
                   + [node(chain[-1], True)])
            for p, q in zip(ids[:-1], ids[1:]):
                if (p, q) not in seen:
                    seen.add((p, q))
                    edges.append((p, q, float(np.hypot(xy[q][0] - xy[p][0],
                                                       xy[q][1] - xy[p][1])), False))

        # Lane changes, outside junctions, where the map allows them.
        tree = cKDTree(np.asarray(xy))
        lane_of = [(w.road_id, w.section_id, w.lane_id) for w in wps]
        for i, w in enumerate(wps):
            if w.is_junction:
                continue
            allowed = str(w.lane_change)
            for side, ok in (('Left', allowed in ('Left', 'Both')),
                             ('Right', allowed in ('Right', 'Both'))):
                if not ok:
                    continue
                n = w.get_left_lane() if side == 'Left' else w.get_right_lane()
                if n is None or n.lane_type != carla.LaneType.Driving \
                        or n.lane_id * w.lane_id <= 0:
                    continue
                target = (n.road_id, n.section_id, n.lane_id)
                nloc = n.transform.location
                for j in tree.query_ball_point((nloc.x, -nloc.y), resolution):
                    if lane_of[j] == target and not wps[j].is_junction:
                        d = float(np.hypot(xy[j][0] - xy[i][0], xy[j][1] - xy[i][1]))
                        edges.append((i, j, d, True))
                        break
        jid = [w.get_junction().id if w.is_junction else -1 for w in wps]
        return cls(xy, edges, jid)

    def neighbors(self, i):
        """[(j, cost)] for every open edge out of node i."""
        return list(self._out[i].items())

    def cost(self, a, b):
        return self._out[a][b]

    def nearest(self, x, y):
        return int(self._tree.query((x, y))[1])

    def close_edges_near(self, x, y, r, record=None):
        """Remove every edge with an end within r meters of (x, y); returns
        how many were removed. With a list as record, appends each removed
        edge (a, b, cost), so reopen can put them back."""
        near = set(self._tree.query_ball_point((x, y), r))
        removed = 0
        for a in range(len(self._out)):
            for b in list(self._out[a]):
                if a in near or b in near:
                    if record is not None:
                        record.append((a, b, self._out[a][b]))
                    del self._out[a][b]
                    removed += 1
        return removed

    def reopen(self, edges):
        """Put back edges removed by close_edges_near."""
        for a, b, c in edges:
            self._out[a][b] = c

    def close_node(self, i):
        """Remove every edge into and out of node i."""
        self._out[i].clear()
        for out in self._out:
            out.pop(i, None)
