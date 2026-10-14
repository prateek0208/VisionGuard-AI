"""
VisionGuard AI - Video Analytics Engine
Module: src/analytics.py
Implements virtual line crossing counters, restricted ROI zone intrusion detectors, and alert logging.
"""

from datetime import datetime
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import pandas as pd


class LineCrossingCounter:
    """
    Virtual tripwire line for bidirectional traffic and pedestrian flow counting.
    Uses cross product to detect when an object trajectory crosses between opposite sides.
    """

    def __init__(self, name: str, p1: Tuple[int, int], p2: Tuple[int, int]):
        self.name = name
        self.p1 = p1
        self.p2 = p2
        self.count_in = 0
        self.count_out = 0
        self.crossed_ids: Dict[int, str] = {}  # track_id -> 'IN' or 'OUT'
        self.class_counts: Dict[str, Dict[str, int]] = {}

    @staticmethod
    def _cross_product(p1: Tuple[int, int], p2: Tuple[int, int], pt: Tuple[int, int]) -> float:
        """Determines which side of the line vector (p1 -> p2) point pt lies on."""
        return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])

    def update(self, tracks) -> List[Dict]:
        """
        Evaluates active tracks to detect crossings.
        Returns list of newly triggered crossing events.
        """
        events = []

        for track in tracks:
            # Need at least two trajectory points to determine crossing
            if len(track.history) < 2:
                continue

            prev_pt = track.history[-2]
            curr_pt = track.history[-1]

            prev_side = self._cross_product(self.p1, self.p2, prev_pt)
            curr_side = self._cross_product(self.p1, self.p2, curr_pt)

            # Check if trajectory crossed the line
            if prev_side * curr_side < 0:
                # Avoid counting the same track multiple times in close succession
                if track.track_id in self.crossed_ids:
                    continue

                direction = "IN" if prev_side > 0 else "OUT"
                if direction == "IN":
                    self.count_in += 1
                else:
                    self.count_out += 1

                self.crossed_ids[track.track_id] = direction

                # Track by class
                c_name = track.class_name
                if c_name not in self.class_counts:
                    self.class_counts[c_name] = {"IN": 0, "OUT": 0}
                self.class_counts[c_name][direction] += 1

                event = {
                    "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "line_name": self.name,
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "direction": direction,
                    "total_in": self.count_in,
                    "total_out": self.count_out
                }
                events.append(event)

        return events

    def reset(self):
        """Resets counts and crossed cache."""
        self.count_in = 0
        self.count_out = 0
        self.crossed_ids.clear()
        self.class_counts.clear()


class RestrictedZone:
    """
    Arbitrary polygon ROI for security perimeter and hazard zone intrusion detection.
    """

    def __init__(self, name: str, polygon: List[Tuple[int, int]], alert_color: Tuple[int, int, int] = (0, 0, 255)):
        self.name = name
        self.polygon = polygon  # List of (x, y) vertices
        self.alert_color = alert_color
        self.active_intrusions: Dict[int, int] = {}  # track_id -> frames inside
        self.total_violations = 0
        self.violation_log: List[Dict] = []

    @staticmethod
    def _point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Ray-casting algorithm to test if a 2D point lies inside a closed polygon."""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def update(self, tracks) -> List[Dict]:
        """
        Checks all active tracks against the polygon.
        Returns list of newly triggered or ongoing zone alerts.
        """
        current_frame_intrusions = set()
        alerts = []

        for track in tracks:
            # Check bottom-center (ground contact point) or centroid
            ground_pt = track.get_bottom_center()
            if self._point_in_polygon(ground_pt, self.polygon):
                current_frame_intrusions.add(track.track_id)

                if track.track_id not in self.active_intrusions:
                    # New intrusion detected
                    self.active_intrusions[track.track_id] = 1
                    self.total_violations += 1
                    alert = {
                        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        "zone_name": self.name,
                        "track_id": track.track_id,
                        "class_name": track.class_name,
                        "severity": "CRITICAL",
                        "event": "ZONE_BREACH"
                    }
                    self.violation_log.append(alert)
                    alerts.append(alert)
                else:
                    self.active_intrusions[track.track_id] += 1

        # Clear tracks that left the zone
        left_zone_ids = set(self.active_intrusions.keys()) - current_frame_intrusions
        for t_id in left_zone_ids:
            del self.active_intrusions[t_id]

        return alerts

    def is_breached(self) -> bool:
        """Returns True if any object is currently inside the zone."""
        return len(self.active_intrusions) > 0

    def reset(self):
        """Resets zone statistics and active intrusions."""
        self.active_intrusions.clear()
        self.total_violations = 0
        self.violation_log.clear()


class AnalyticsEngine:
    """
    Central hub managing lines, zones, metrics, and report generation.
    """

    def __init__(self):
        self.lines: Dict[str, LineCrossingCounter] = {}
        self.zones: Dict[str, RestrictedZone] = {}
        self.all_seen_ids: Set[int] = set()
        self.class_distribution: Dict[str, int] = {}
        self.event_log: List[Dict] = []

    def add_line(self, name: str, p1: Tuple[int, int], p2: Tuple[int, int]) -> LineCrossingCounter:
        """Registers a new line counter."""
        line = LineCrossingCounter(name, p1, p2)
        self.lines[name] = line
        return line

    def add_zone(self, name: str, polygon: List[Tuple[int, int]]) -> RestrictedZone:
        """Registers a new restricted zone."""
        zone = RestrictedZone(name, polygon)
        self.zones[name] = zone
        return zone

    def process_frame(self, tracks) -> Dict:
        """Processes a frame's active tracks and returns summary events."""
        for t in tracks:
            if t.track_id not in self.all_seen_ids:
                self.all_seen_ids.add(t.track_id)
                self.class_distribution[t.class_name] = self.class_distribution.get(t.class_name, 0) + 1

        frame_events = []

        # Process lines
        for line in self.lines.values():
            crossing_events = line.update(tracks)
            frame_events.extend(crossing_events)
            self.event_log.extend(crossing_events)

        # Process zones
        for zone in self.zones.values():
            zone_alerts = zone.update(tracks)
            frame_events.extend(zone_alerts)
            self.event_log.extend(zone_alerts)

        return {
            "total_unique_objects": len(self.all_seen_ids),
            "currently_active": len(tracks),
            "class_distribution": dict(self.class_distribution),
            "events": frame_events
        }

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Converts event log into a structured pandas DataFrame."""
        if not self.event_log:
            return pd.DataFrame(columns=["timestamp", "track_id", "class_name", "type", "details"])

        formatted_rows = []
        for e in self.event_log:
            if "line_name" in e:
                formatted_rows.append({
                    "timestamp": e.get("timestamp"),
                    "track_id": e.get("track_id"),
                    "class_name": e.get("class_name"),
                    "type": "LINE_CROSSING",
                    "details": f"{e.get('line_name')}: {e.get('direction')}"
                })
            else:
                formatted_rows.append({
                    "timestamp": e.get("timestamp"),
                    "track_id": e.get("track_id"),
                    "class_name": e.get("class_name"),
                    "type": "ZONE_INTRUSION",
                    "details": f"{e.get('zone_name')} [BREACH]"
                })

        return pd.DataFrame(formatted_rows)

    def reset(self):
        """Resets all lines, zones, and counters."""
        for line in self.lines.values():
            line.reset()
        for zone in self.zones.values():
            zone.reset()
        self.all_seen_ids.clear()
        self.class_distribution.clear()
        self.event_log.clear()
