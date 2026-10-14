"""
VisionGuard AI - Multi-Object Tracker (MOT)
Module: src/tracker.py
Implements an IoU + Centroid Distance tracking engine with trajectory history and occlusion handling.
"""

from collections import deque
from typing import List, Dict, Tuple, Optional
import numpy as np


class TrackedObject:
    """Represents a single tracked target across video frames."""

    def __init__(self, track_id: int, bbox: List[int], class_name: str, confidence: float, max_history: int = 30):
        self.track_id = track_id
        self.bbox = [int(v) for v in bbox]  # [x1, y1, x2, y2]
        self.class_name = class_name
        self.confidence = float(confidence)
        self.history = deque(maxlen=max_history)
        self.lost_frames = 0
        self.is_active = True
        self.velocity = (0.0, 0.0)

        # Update initial centroid and history
        cx, cy = self.get_centroid()
        self.history.append((cx, cy))

    def get_centroid(self) -> Tuple[int, int]:
        """Calculates center coordinate of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def get_bottom_center(self) -> Tuple[int, int]:
        """Calculates the ground-contact point (bottom-center) of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2), int(y2)

    def update(self, bbox: List[int], confidence: float):
        """Updates object state with newly matched detection."""
        prev_cx, prev_cy = self.get_centroid()
        self.bbox = [int(v) for v in bbox]
        self.confidence = float(confidence)
        self.lost_frames = 0
        self.is_active = True

        new_cx, new_cy = self.get_centroid()
        self.velocity = (new_cx - prev_cx, new_cy - prev_cy)
        self.history.append((new_cx, new_cy))

    def mark_missed(self):
        """Marks object as missed in the current frame."""
        self.lost_frames += 1
        # Predict next position based on velocity if available
        if len(self.history) >= 2:
            vx, vy = self.velocity
            x1, y1, x2, y2 = self.bbox
            self.bbox = [int(x1 + vx), int(y1 + vy), int(x2 + vx), int(y2 + vy)]
            cx, cy = self.get_centroid()
            self.history.append((cx, cy))


class MultiObjectTracker:
    """
    Real-time Multi-Object Tracker.
    Uses a hybrid metric of spatial IoU and Euclidean distance for robust object association.
    """

    def __init__(self, max_lost: int = 15, iou_threshold: float = 0.25, max_distance: float = 120.0):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedObject] = {}

    @staticmethod
    def compute_iou(box_a: List[int], box_b: List[int]) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_w = max(0, x_b - x_a)
        inter_h = max(0, y_b - y_a)
        inter_area = inter_w * inter_h

        if inter_area == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = float(area_a + area_b - inter_area)

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def compute_distance(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
        """Computes Euclidean distance between two 2D points."""
        return float(np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]))

    def update(self, detections: List[Dict]) -> List[TrackedObject]:
        """
        Updates active tracks with incoming detections.
        Each detection is a dict with keys: 'bbox' [x1, y1, x2, y2], 'class_name', 'confidence'.
        """
        active_track_ids = list(self.tracks.keys())

        # If no active tracks, initialize all detections as new tracks
        if not active_track_ids:
            for det in detections:
                new_track = TrackedObject(
                    track_id=self.next_track_id,
                    bbox=det["bbox"],
                    class_name=det.get("class_name", "object"),
                    confidence=det.get("confidence", 1.0)
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
            return list(self.tracks.values())

        # If no detections in current frame, mark all active tracks as missed
        if not detections:
            expired_ids = []
            for t_id, track in self.tracks.items():
                track.mark_missed()
                if track.lost_frames > self.max_lost:
                    expired_ids.append(t_id)
            for t_id in expired_ids:
                del self.tracks[t_id]
            return [t for t in self.tracks.values() if t.lost_frames == 0]

        # Calculate cost matrix between existing tracks and incoming detections
        unmatched_tracks = set(active_track_ids)
        unmatched_dets = set(range(len(detections)))
        matches: List[Tuple[int, int]] = []

        # First pass: Match by high IoU overlap
        for det_idx in list(unmatched_dets):
            det_box = detections[det_idx]["bbox"]
            best_track_id = None
            best_iou = self.iou_threshold

            for t_id in unmatched_tracks:
                track = self.tracks[t_id]
                iou = self.compute_iou(track.bbox, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = t_id

            if best_track_id is not None:
                matches.append((best_track_id, det_idx))
                unmatched_tracks.remove(best_track_id)
                unmatched_dets.remove(det_idx)

        # Second pass: Match remaining tracks by Centroid Euclidean distance
        for det_idx in list(unmatched_dets):
            det_box = detections[det_idx]["bbox"]
            det_center = ((det_box[0] + det_box[2]) // 2, (det_box[1] + det_box[3]) // 2)
            best_track_id = None
            min_dist = self.max_distance

            for t_id in unmatched_tracks:
                track = self.tracks[t_id]
                dist = self.compute_distance(track.get_centroid(), det_center)
                if dist < min_dist:
                    min_dist = dist
                    best_track_id = t_id

            if best_track_id is not None:
                matches.append((best_track_id, det_idx))
                unmatched_tracks.remove(best_track_id)
                unmatched_dets.remove(det_idx)

        # Update matched tracks
        for t_id, det_idx in matches:
            det = detections[det_idx]
            self.tracks[t_id].update(bbox=det["bbox"], confidence=det.get("confidence", 1.0))
            if "class_name" in det:
                self.tracks[t_id].class_name = det["class_name"]

        # Increment lost count for unmatched tracks and prune dead tracks
        expired_ids = []
        for t_id in unmatched_tracks:
            self.tracks[t_id].mark_missed()
            if self.tracks[t_id].lost_frames > self.max_lost:
                expired_ids.append(t_id)

        for t_id in expired_ids:
            del self.tracks[t_id]

        # Register new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = TrackedObject(
                track_id=self.next_track_id,
                bbox=det["bbox"],
                class_name=det.get("class_name", "object"),
                confidence=det.get("confidence", 1.0)
            )
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        # Return only currently active (not lost) tracks for visualization
        return [t for t in self.tracks.values() if t.lost_frames == 0]

    def reset(self):
        """Clears all tracks and resets ID counter."""
        self.tracks.clear()
        self.next_track_id = 1
