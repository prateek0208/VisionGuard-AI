"""
VisionGuard AI - Command Line Interface (CLI) Pipeline
Module: cli.py
Processes video files, streams, or images through the VisionGuard AI detection, tracking, and analytics pipeline.
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

from src.detector import ObjectDetector
from src.tracker import MultiObjectTracker
from src.analytics import AnalyticsEngine
from src.visualizer import Visualizer
from src.utils import FPSCounter, create_video_writer


def run_pipeline(
    source: str,
    output_path: Optional[str] = None,
    report_path: Optional[str] = None,
    mode: str = "motion",
    confidence: float = 0.40,
    nms_thresh: float = 0.35,
    draw_trails: bool = True,
    draw_hud: bool = True,
    max_frames: Optional[int] = None
):
    """Executes the full VisionGuard pipeline on a video or image source."""
    # Ensure source exists
    if not os.path.exists(source) and not source.isdigit():
        print(f"[ERROR] Source '{source}' not found.")
        sys.exit(1)

    # Initialize components
    detector = ObjectDetector(mode=mode, confidence_threshold=confidence, nms_threshold=nms_thresh)
    tracker = MultiObjectTracker(max_lost=12, iou_threshold=0.25, max_distance=100.0)
    analytics = AnalyticsEngine()
    visualizer = Visualizer(draw_trails=draw_trails, draw_hud=draw_hud)
    fps_counter = FPSCounter()

    # Configure default tripwire line and restricted hazard zone
    analytics.add_line("Main-Tripwire", (400, 120), (400, 420))
    analytics.add_zone("Restricted-Area-A", [(80, 140), (300, 140), (300, 240), (80, 240)])

    # Check if input is image
    is_image = source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    if is_image:
        print(f"[*] Processing static image: {source}")
        frame = cv2.imread(source)
        if frame is None:
            print("[ERROR] Failed to read image.")
            return

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        summary = analytics.process_frame(tracks)

        # Render visualizations
        annotated = frame.copy()
        annotated = visualizer.draw_zones(annotated, analytics.zones)
        annotated = visualizer.draw_lines(annotated, analytics.lines)
        annotated = visualizer.draw_tracks(annotated, tracks)
        annotated = visualizer.draw_hud(
            annotated,
            fps=30.0,
            active_count=len(tracks),
            total_unique=summary["total_unique_objects"],
            alert_count=len(summary["events"])
        )

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            cv2.imwrite(output_path, annotated)
            print(f"[OK] Saved annotated image to: {output_path}")

        print(f"[SUMMARY] Detections: {len(detections)}, Active Tracks: {len(tracks)}")
        return

    # Process video
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = create_video_writer(output_path, fps, width, height)

    frame_idx = 0
    print(f"[*] Commencing video analytics on '{source}' ({width}x{height} @ {fps:.1f} FPS)...")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        current_fps = fps_counter.tick()

        # 1. Detection
        detections = detector.detect(frame)

        # 2. Tracking
        tracks = tracker.update(detections)

        # 3. Analytics
        summary = analytics.process_frame(tracks)

        # 4. Visualization
        annotated = frame.copy()
        annotated = visualizer.draw_zones(annotated, analytics.zones)
        annotated = visualizer.draw_lines(annotated, analytics.lines)
        annotated = visualizer.draw_tracks(annotated, tracks)
        annotated = visualizer.draw_hud(
            annotated,
            fps=current_fps,
            active_count=len(tracks),
            total_unique=summary["total_unique_objects"],
            alert_count=len(analytics.event_log)
        )

        if writer:
            writer.write(annotated)

        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx} | FPS: {current_fps:.1f} | Active: {len(tracks)} | Total Unique: {summary['total_unique_objects']} | Events: {len(analytics.event_log)}")

    cap.release()
    if writer:
        writer.release()
        print(f"[OK] Saved annotated video to: {output_path}")

    # Export report
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        df = analytics.get_summary_dataframe()
        df.to_csv(report_path, index=False)
        print(f"[OK] Exported analytics log to: {report_path} ({len(df)} records)")

    print("[FINISHED] Video analytics run complete.")


def main():
    parser = argparse.ArgumentParser(description="VisionGuard AI - Video Analytics Pipeline")
    parser.add_argument("--source", type=str, default="data/samples/sample_traffic.mp4", help="Video/image source path or webcam index (0)")
    parser.add_argument("--output", type=str, default="output/annotated_traffic.mp4", help="Output annotated media path")
    parser.add_argument("--report", type=str, default="output/analytics_report.csv", help="Output CSV report path")
    parser.add_argument("--mode", type=str, default="motion", choices=["motion", "dnn"], help="Detection mode")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-trails", action="store_true", help="Disable trajectory trails")
    parser.add_argument("--no-hud", action="store_true", help="Disable top telemetry HUD bar")

    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        output_path=args.output,
        report_path=args.report,
        mode=args.mode,
        confidence=args.conf,
        draw_trails=not args.no_trails,
        draw_hud=not args.no_hud,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    from typing import Optional
    main()
