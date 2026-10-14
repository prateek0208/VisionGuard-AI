"""
Unit Tests for Multi-Object Tracker (MOT)
Module: tests/test_tracker.py
"""

import pytest
from src.tracker import MultiObjectTracker, TrackedObject


def test_iou_identical_boxes():
    box = [10, 10, 50, 50]
    iou = MultiObjectTracker.compute_iou(box, box)
    assert pytest.approx(iou, 0.001) == 1.0


def test_iou_disjoint_boxes():
    box_a = [0, 0, 10, 10]
    box_b = [20, 20, 30, 30]
    iou = MultiObjectTracker.compute_iou(box_a, box_b)
    assert iou == 0.0


def test_iou_partial_overlap():
    box_a = [0, 0, 20, 20]     # Area = 400
    box_b = [10, 0, 30, 20]    # Area = 400, Intersection = 10*20 = 200, Union = 600
    iou = MultiObjectTracker.compute_iou(box_a, box_b)
    assert pytest.approx(iou, 0.01) == 200.0 / 600.0


def test_track_initialization_and_persistence():
    tracker = MultiObjectTracker(max_lost=5, iou_threshold=0.3)

    # Frame 1: Single detection
    dets_f1 = [{"bbox": [50, 50, 100, 100], "class_name": "car", "confidence": 0.9}]
    tracks_f1 = tracker.update(dets_f1)

    assert len(tracks_f1) == 1
    t1_id = tracks_f1[0].track_id
    assert tracks_f1[0].class_name == "car"

    # Frame 2: Object moved slightly
    dets_f2 = [{"bbox": [54, 52, 104, 102], "class_name": "car", "confidence": 0.92}]
    tracks_f2 = tracker.update(dets_f2)

    assert len(tracks_f2) == 1
    # ID must be preserved
    assert tracks_f2[0].track_id == t1_id


def test_track_expiration():
    tracker = MultiObjectTracker(max_lost=2)
    dets = [{"bbox": [50, 50, 100, 100], "class_name": "person", "confidence": 0.85}]
    tracker.update(dets)
    assert len(tracker.tracks) == 1

    # Miss 1 frame
    tracker.update([])
    assert len(tracker.tracks) == 1

    # Miss 2 frames
    tracker.update([])
    assert len(tracker.tracks) == 1

    # Miss 3 frames (> max_lost=2), track should be deleted
    tracker.update([])
    assert len(tracker.tracks) == 0
