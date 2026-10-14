"""
Unit Tests for Video Analytics Engine
Module: tests/test_analytics.py
"""

from src.analytics import LineCrossingCounter, RestrictedZone, AnalyticsEngine
from src.tracker import TrackedObject


def test_line_crossing_in():
    # Vertical line at x = 100, extending from y = 0 to y = 200
    line = LineCrossingCounter("Gate-1", (100, 0), (100, 200))

    track = TrackedObject(track_id=1, bbox=[80, 50, 95, 70], class_name="person", confidence=0.9)
    # Simulate movement from left (x ~ 87) to right (x ~ 115)
    line.update([track])
    assert line.count_in == 0

    # Next frame crosses x = 100
    track.update([110, 50, 125, 70], confidence=0.9)
    events = line.update([track])

    assert len(events) == 1
    assert line.count_in + line.count_out == 1


def test_point_in_polygon():
    # Square zone from (10, 10) to (50, 50)
    polygon = [(10, 10), (50, 10), (50, 50), (10, 50)]

    assert RestrictedZone._point_in_polygon((30, 30), polygon) is True
    assert RestrictedZone._point_in_polygon((5, 5), polygon) is False
    assert RestrictedZone._point_in_polygon((60, 30), polygon) is False


def test_zone_intrusion_alert():
    zone = RestrictedZone("Danger-Zone", [(10, 10), (100, 10), (100, 100), (10, 100)])

    # Target outside
    track = TrackedObject(track_id=1, bbox=[150, 150, 180, 180], class_name="person", confidence=0.9)
    alerts = zone.update([track])
    assert len(alerts) == 0
    assert not zone.is_breached()

    # Target steps inside
    track.update([30, 30, 60, 60], confidence=0.9)
    alerts = zone.update([track])
    assert len(alerts) == 1
    assert alerts[0]["event"] == "ZONE_BREACH"
    assert zone.is_breached()
