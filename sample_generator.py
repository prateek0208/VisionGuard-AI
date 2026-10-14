"""
VisionGuard AI - Synthetic Benchmark & Sample Footage Generator
Module: sample_generator.py
Generates simulated traffic/pedestrian video clips and test images for instant evaluation without external downloads.
"""

import os
from typing import Tuple
import cv2
import numpy as np


def draw_vehicle(img: np.ndarray, x: int, y: int, color: Tuple[int, int, int], is_truck: bool = False):
    """Draws a synthetic vehicle (car or truck) with wheels and windows."""
    w, h = (90, 48) if is_truck else (65, 34)

    # Chassis
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), 2)

    # Cabin / Roof
    cabin_w = int(w * 0.5)
    cabin_h = int(h * 0.45)
    cv2.rectangle(img, (x + 12, y + 5), (x + 12 + cabin_w, y + 5 + cabin_h), (210, 230, 240), -1)

    # Wheels
    wheel_r = 7 if is_truck else 5
    cv2.circle(img, (x + 15, y + h), wheel_r, (20, 20, 20), -1)
    cv2.circle(img, (x + w - 15, y + h), wheel_r, (20, 20, 20), -1)
    if is_truck:
        cv2.circle(img, (x + w - 30, y + h), wheel_r, (20, 20, 20), -1)


def draw_pedestrian(img: np.ndarray, x: int, y: int, color: Tuple[int, int, int]):
    """Draws a synthetic pedestrian figure."""
    # Head
    cv2.circle(img, (x + 10, y + 10), 7, (240, 200, 160), -1)
    # Torso
    cv2.rectangle(img, (x + 5, y + 18), (x + 15, y + 42), color, -1)
    # Legs
    cv2.line(img, (x + 7, y + 42), (x + 5, y + 58), (40, 40, 40), 3)
    cv2.line(img, (x + 13, y + 42), (x + 15, y + 58), (40, 40, 40), 3)


def generate_benchmark_video(
    output_path: str = "data/samples/sample_traffic.mp4",
    num_frames: int = 150,
    width: int = 800,
    height: int = 480,
    fps: int = 30
):
    """Generates an MP4 video showcasing road traffic, pedestrians, and perimeter zones."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Object trajectories
    # Vehicle 1 (Car - moving right)
    v1_x, v1_y, v1_speed = -60, 260, 5
    # Vehicle 2 (Truck - moving left)
    v2_x, v2_y, v2_speed = width + 50, 340, -4
    # Pedestrian 1 (walking across)
    p1_x, p1_y, p1_vy = 380, 480, -3
    # Vehicle 3 (Car entering restricted zone)
    v3_x, v3_y, v3_speed = -120, 180, 6

    for f in range(num_frames):
        # Create road scene background
        frame = np.full((height, width, 3), (70, 75, 80), dtype=np.uint8)

        # Sidewalk (top)
        cv2.rectangle(frame, (0, 0), (width, 120), (130, 135, 140), -1)
        # Grass verge
        cv2.rectangle(frame, (0, 0), (width, 60), (50, 110, 50), -1)

        # Road lanes
        # Lane divider (dashed white line)
        for dx in range(0, width, 40):
            cv2.line(frame, (dx, 220), (dx + 25, 220), (255, 255, 255), 2)
            cv2.line(frame, (dx, 310), (dx + 25, 310), (255, 255, 255), 2)
        # Yellow central divider
        cv2.line(frame, (0, 310), (width, 310), (0, 220, 255), 3)

        # Sidewalk (bottom)
        cv2.rectangle(frame, (0, 420), (width, height), (130, 135, 140), -1)

        # Crosswalk markings
        for dy in range(130, 410, 30):
            cv2.rectangle(frame, (350, dy), (410, dy + 18), (240, 240, 240), -1)

        # Render moving actors
        # Car 1
        draw_vehicle(frame, int(v1_x), int(v1_y), (40, 80, 220), is_truck=False)
        v1_x += v1_speed

        # Truck 1
        draw_vehicle(frame, int(v2_x), int(v2_y), (180, 100, 30), is_truck=True)
        v2_x += v2_speed

        # Pedestrian 1
        draw_pedestrian(frame, int(p1_x), int(p1_y), (200, 40, 40))
        p1_y += p1_vy

        # Car 3
        if f > 20:
            draw_vehicle(frame, int(v3_x), int(v3_y), (30, 180, 50), is_truck=False)
            v3_x += v3_speed

        out.write(frame)

    out.release()
    print(f"[OK] Generated benchmark video at: {output_path}")


def generate_benchmark_image(output_path: str = "data/samples/sample_frame.jpg"):
    """Generates a static benchmark image containing vehicles and pedestrians."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 800, 480
    frame = np.full((height, width, 3), (70, 75, 80), dtype=np.uint8)

    # Road lanes
    cv2.rectangle(frame, (0, 0), (width, 120), (130, 135, 140), -1)
    cv2.rectangle(frame, (0, 420), (width, height), (130, 135, 140), -1)

    for dx in range(0, width, 40):
        cv2.line(frame, (dx, 220), (dx + 25, 220), (255, 255, 255), 2)
    cv2.line(frame, (0, 310), (width, 310), (0, 220, 255), 3)

    # Draw multiple vehicles and pedestrians
    draw_vehicle(frame, 180, 250, (40, 80, 220), is_truck=False)
    draw_vehicle(frame, 450, 340, (180, 100, 30), is_truck=True)
    draw_vehicle(frame, 320, 150, (30, 180, 50), is_truck=False)
    draw_pedestrian(frame, 380, 220, (200, 40, 40))
    draw_pedestrian(frame, 620, 70, (20, 160, 220))

    cv2.imwrite(output_path, frame)
    print(f"[OK] Generated benchmark image at: {output_path}")


if __name__ == "__main__":
    generate_benchmark_video()
    generate_benchmark_image()
