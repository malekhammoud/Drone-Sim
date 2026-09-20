"""Tests for the verified detector pipeline (Step 1 color + Step 2 CNN + Step 3 temporal).

The Stage-2 model is trained on real sim frames, so a synthetic cartoon boat is
deliberately rejected by it. We therefore test each stage on its own terms:
Stage 1 must find the synthetic boat, the pipeline must run with and without the
model, and the temporal filter must be able to confirm a track over frames.
"""
import os
import unittest

import cv2
import numpy as np

from tools.detect_color import ColorAnomalyDetector
from tools.detect_verified import VerifiedDetector

MODEL = "models/patch_verifier.pt"


def _synthetic_scene() -> np.ndarray:
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:, :] = [70, 40, 15]                       # dark blue water
    cv2.circle(img, (260, 140), 50, (220, 225, 225), -1)   # iceberg
    cv2.rectangle(img, (110, 85), (130, 95), (25, 30, 200), -1)   # red hull
    cv2.rectangle(img, (116, 88), (124, 92), (150, 150, 150), -1)  # cabin
    return img


class TestVerifiedDetector(unittest.TestCase):
    def setUp(self):
        self.model_path = MODEL

    def test_end_to_end_real_frame(self):
        if not os.path.exists(self.model_path):
            self.skipTest(f"{self.model_path} not found")
        frame_path = "patrol_run/2026-09-19T21-13-42/frames/00600.jpg"
        if not os.path.exists(frame_path):
            self.skipTest(f"{frame_path} not found")

        detector = VerifiedDetector(model_path=self.model_path, min_color_score=0.25, min_verify_prob=0.30)
        img = cv2.imread(frame_path)
        detections = detector.detect(img)
        self.assertGreaterEqual(len(detections), 1, "Should detect vessel in real frame")
        top = detections[0]
        self.assertTrue(top.contains(313.8, 264.4, margin=15.0))
        self.assertGreater(top.score, 0.50)

    def test_stage1_finds_synthetic_boat(self):
        det = ColorAnomalyDetector(min_area=2, max_area=800, min_score=0.25)
        cands = det.detect(_synthetic_scene())
        self.assertGreaterEqual(len(cands), 1)
        self.assertTrue(cands[0].contains(120, 90, margin=10.0))

    def test_pipeline_runs_stage1_only(self):
        det = VerifiedDetector(model_path=None, min_color_score=0.25,
                               min_verify_prob=0.30, enable_temporal=False)
        dets = det.detect(_synthetic_scene())
        self.assertGreaterEqual(len(dets), 1, "Stage 1 only should keep the synthetic boat")

    def test_temporal_filter_confirms_over_frames(self):
        # A persistent candidate should become a confirmed track after min_hits.
        det = VerifiedDetector(model_path=None, min_color_score=0.25,
                               min_verify_prob=0.30, enable_temporal=True, min_hits=3)
        confirmed = False
        for i in range(4):
            dets = det.detect(_synthetic_scene(), frame_idx=i)
            if dets and getattr(dets[0], "is_confirmed", False):
                confirmed = True
                break
        self.assertTrue(confirmed, "persistent candidate should confirm within 4 frames")

    def test_full_pipeline_runs_with_model(self):
        if not os.path.exists(MODEL):
            self.skipTest(f"{MODEL} not found")
        det = VerifiedDetector(model_path=MODEL, min_color_score=0.25,
                               min_verify_prob=0.30)
        # Should run without error and return a list (the real model may reject
        # this synthetic patch — that is correct behaviour).
        self.assertIsInstance(det.detect(_synthetic_scene()), list)


if __name__ == "__main__":
    unittest.main()
