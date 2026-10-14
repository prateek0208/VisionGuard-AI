"""
VisionGuard AI - Utility Functions
Module: src/utils.py
Provides FPS estimation, video I/O, file exports, and geometry helpers.
"""

import time
from collections import deque
from typing import Optional, Tuple
import cv2
import numpy as np


class FPSCounter:
    """Calculates smoothed frames-per-second (FPS)."""

    def __init__(self, window_size: int = 30):
        self.timestamps = deque(maxlen=window_size)

    def tick(self) -> float:
        """Call this on each frame iteration. Returns smoothed FPS."""
        now = time.time()
        self.timestamps.append(now)

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0

        return float(len(self.timestamps) - 1) / elapsed


def create_video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
    fourcc_str: str = "mp4v"
) -> cv2.VideoWriter:
    """Creates an OpenCV VideoWriter instance."""
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
