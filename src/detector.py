"""
VisionGuard AI - Object Detection Engine
Module: src/detector.py
Provides modular object detection with support for Motion/Contour background subtraction,
OpenCV DNN deep learning models, and automated synthetic test generators.
"""

from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np


class ObjectDetector:
    """
    Modular detector supporting adaptive Computer Vision background subtraction (MOG2),
    color-motion segmentation, and OpenCV DNN deep learning models.
    """

    COCO_CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]

    def __init__(
        self,
        mode: str = "motion",
        confidence_threshold: float = 0.40,
        nms_threshold: float = 0.35,
        min_area: int = 600,
        max_area: int = 150000
    ):
        self.mode = mode.lower()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_area = min_area
        self.max_area = max_area

        # Adaptive Background Subtractor (MOG2) with shadow detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
            detectShadows=True
        )

        # Morphological kernels for noise cleanup and hole filling
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # Optional DNN net
        self.net = None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Runs object detection on the input BGR image/frame.
        Returns list of detection dictionaries:
        [{'bbox': [x1, y1, x2, y2], 'class_name': str, 'confidence': float}]
        """
        if frame is None or frame.size == 0:
            return []

        if self.mode == "motion":
            return self._detect_motion(frame)
        elif self.mode == "dnn" and self.net is not None:
            return self._detect_dnn(frame)
        else:
            return self._detect_motion(frame)

    def _detect_motion(self, frame: np.ndarray) -> List[Dict]:
        """High-speed adaptive motion and contour-based object detection."""
        # 1. Preprocess: Blur to reduce camera sensor noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # 2. Compute foreground mask with MOG2
        fg_mask = self.bg_subtractor.apply(blurred)

        # 3. Filter out shadows (gray pixels 127 in MOG2)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # 4. Morphological opening (remove specks) and closing (bridge broken contours)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel_close)

        # 5. Extract external contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes: List[List[int]] = []
        confidences: List[float] = []
        class_names: List[str] = []

        h_img, w_img = frame.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Heuristic aspect ratio and size classification
            aspect_ratio = h / float(w)

            # Class identification based on aspect ratio & footprint
            if aspect_ratio >= 1.6 and h > 45:
                label = "person"
            elif aspect_ratio < 1.1 and area > 4000:
                label = "truck" if area > 18000 else "car"
            elif 1.1 <= aspect_ratio < 1.6:
                label = "bicycle" if area < 4500 else "car"
            else:
                label = "vehicle"

            # Confidence estimated from contour solidity & size
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0.5
            confidence = min(0.98, max(0.45, solidity * 0.95))

            if confidence >= self.confidence_threshold:
                raw_boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_names.append(label)

        # 6. Non-Maximum Suppression (NMS) to eliminate duplicate boxes
        if not raw_boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            bboxes=raw_boxes,
            scores=confidences,
            score_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = raw_boxes[i]
                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "class_name": class_names[i],
                    "confidence": round(float(confidences[i]), 2)
                })

        return detections

    def _detect_dnn(self, frame: np.ndarray) -> List[Dict]:
        """DNN-based forward pass for deep neural network models."""
        h_img, w_img = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False
        )
        self.net.setInput(blob)
        dnn_output = self.net.forward()

        detections = []
        for i in range(dnn_output.shape[2]):
            confidence = float(dnn_output[0, 0, i, 2])
            if confidence > self.confidence_threshold:
                idx = int(dnn_output[0, 0, i, 1])
                class_name = self.COCO_CLASSES[idx] if idx < len(self.COCO_CLASSES) else "object"

                box = dnn_output[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
                x1, y1, x2, y2 = box.astype("int")

                detections.append({
                    "bbox": [max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)],
                    "class_name": class_name,
                    "confidence": round(confidence, 2)
                })

        return detections

    def reset(self):
        """Resets the background subtractor history."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
            detectShadows=True
        )
