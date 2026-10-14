"""
VisionGuard AI - Computer Vision Visualizer & HUD Engine
Module: src/visualizer.py
Renders high-contrast bounding boxes, trajectory trails, line counters,
restricted zone polygon overlays, and telemetry HUD metrics.
"""

from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np


class Visualizer:
    """Renders professional overlays on video frames."""

    CLASS_COLORS = {
        "person": (255, 191, 0),     # Deep Sky Blue (BGR)
        "car": (50, 205, 50),        # Lime Green
        "truck": (0, 140, 255),      # Orange
        "bicycle": (211, 0, 148),    # Violet / Magenta
        "vehicle": (0, 215, 255),    # Gold / Yellow
        "default": (200, 200, 200)   # Neutral Gray
    }

    def __init__(self, draw_trails: bool = True, draw_hud: bool = True):
        self.show_trails = draw_trails
        self.show_hud = draw_hud

    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Returns BGR color tuple for a given object class."""
        return self.CLASS_COLORS.get(class_name.lower(), self.CLASS_COLORS["default"])

    def draw_corner_rect(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        corner_len: int = 15
    ):
        """Draws modern corner brackets around the bounding box."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Base thin rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Top-Left corner
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)

        # Top-Right corner
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)

        # Bottom-Left corner
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)

        # Bottom-Right corner
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)

    def draw_tracks(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        """Renders bounding boxes, badges, and trajectory trails for tracked targets."""
        for track in tracks:
            color = self.get_color(track.class_name)
            x1, y1, x2, y2 = track.bbox

            # Draw trajectory trail
            if self.show_trails and len(track.history) > 1:
                pts = list(track.history)
                for i in range(1, len(pts)):
                    alpha = float(i) / len(pts)
                    thick = max(1, int(3 * alpha))
                    cv2.line(frame, pts[i - 1], pts[i], color, thick, cv2.LINE_AA)

            # Draw modern bounding box
            self.draw_corner_rect(frame, (x1, y1), (x2, y2), color, thickness=3)

            # Draw label badge
            label = f"#{track.track_id} {track.class_name.upper()} {int(track.confidence * 100)}%"
            font_scale = 0.5
            font_thick = 1
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = y1
            cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 10, badge_y2), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thick,
                cv2.LINE_AA
            )

        return frame

    def draw_lines(self, frame: np.ndarray, lines: dict) -> np.ndarray:
        """Renders virtual counting lines with count badges."""
        for name, line in lines.items():
            p1 = line.p1
            p2 = line.p2
            # Draw line
            cv2.line(frame, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(frame, p1, 5, (0, 200, 255), -1)
            cv2.circle(frame, p2, 5, (0, 200, 255), -1)

            # Draw label and counts near midpoint
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2

            text = f"LINE [{name}]: IN {line.count_in} | OUT {line.count_out}"
            cv2.rectangle(frame, (mid_x - 10, mid_y - 20), (mid_x + 230, mid_y + 10), (20, 20, 20), -1)
            cv2.rectangle(frame, (mid_x - 10, mid_y - 20), (mid_x + 230, mid_y + 10), (0, 255, 255), 1)
            cv2.putText(
                frame,
                text,
                (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return frame

    def draw_zones(self, frame: np.ndarray, zones: dict) -> np.ndarray:
        """Renders semi-transparent restricted polygon zones and violation banners."""
        overlay = frame.copy()
        alpha = 0.35

        for name, zone in zones.items():
            pts = np.array(zone.polygon, np.int32).reshape((-1, 1, 2))
            is_alert = zone.is_breached()

            fill_color = (0, 0, 220) if is_alert else (0, 180, 0)
            border_color = (0, 0, 255) if is_alert else (0, 255, 0)

            # Fill polygon on overlay
            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

            # Zone label
            centroid_x = int(np.mean([p[0] for p in zone.polygon]))
            centroid_y = int(np.mean([p[1] for p in zone.polygon]))
            status_text = f"RESTRICTED: {name} (VIOLATIONS: {zone.total_violations})"
            if is_alert:
                status_text = f"!! BREACH: {name} !!"

            cv2.putText(
                frame,
                status_text,
                (centroid_x - 60, centroid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # Blend semi-transparent overlay
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_hud(self, frame: np.ndarray, fps: float, active_count: int, total_unique: int, alert_count: int) -> np.ndarray:
        """Renders top HUD status bar with real-time analytics telemetry."""
        if not self.show_hud:
            return frame

        h, w = frame.shape[:2]
        hud_h = 42

        # Semi-transparent dark bar at the top
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (15, 15, 18), -1)
        cv2.addWeighted(hud_overlay, 0.85, frame, 0.15, 0, frame)

        # Draw bottom separator line
        cv2.line(frame, (0, hud_h), (w, hud_h), (60, 60, 60), 1)

        # Telemetry fields
        stats = [
            ("VisionGuard AI", (0, 215, 255)),
            (f"FPS: {fps:.1f}", (50, 255, 50)),
            (f"Active Targets: {active_count}", (255, 255, 255)),
            (f"Total Unique: {total_unique}", (255, 200, 0)),
            (f"Alerts: {alert_count}", (0, 69, 255) if alert_count > 0 else (120, 120, 120))
        ]

        x_offset = 20
        y_text = 27
        for label, color in stats:
            cv2.putText(frame, label, (x_offset, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
            tw, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
            x_offset += tw + 30

        return frame
