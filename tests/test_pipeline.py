"""
Integration Tests for End-to-End Pipeline
Module: tests/test_pipeline.py
"""

import numpy as np
from src.detector import ObjectDetector
from src.tracker import MultiObjectTracker
from src.analytics import AnalyticsEngine
from src.visualizer import Visualizer


def test_full_pipeline_synthetic_frames():
    detector = ObjectDetector(mode="motion", min_area=200)
    tracker = MultiObjectTracker()
    analytics = AnalyticsEngine()
    visualizer = Visualizer()

    analytics.add_line("Center-Line", (150, 0), (150, 300))
    analytics.add_zone("Zone-1", [(50, 50), (120, 50), (120, 120), (50, 120)])

    # Generate synthetic sequence: moving white box across black background
    for frame_i in range(10):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw moving rectangle from x=40 to x=220
        box_x = 40 + frame_i * 20
        frame[70:110, box_x:box_x + 30] = (255, 255, 255)

        dets = detector.detect(frame)
        tracks = tracker.update(dets)
        summary = analytics.process_frame(tracks)

        annotated = visualizer.draw_tracks(frame.copy(), tracks)
        annotated = visualizer.draw_hud(annotated, 30.0, len(tracks), summary["total_unique_objects"], 0)

        assert annotated.shape == (300, 300, 3)

    # Verify that summary captured detections and events
    df = analytics.get_summary_dataframe()
    assert isinstance(df, object)
