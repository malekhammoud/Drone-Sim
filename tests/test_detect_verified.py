"""Tests for Step 2: Verified Detector (Step 1 + Step 2 pipeline)."""
import unittest
import numpy as np
import cv2
import os

from tools.detect_verified import VerifiedDetector


class TestVerifiedDetector(unittest.TestCase):
    def setUp(self):
        self.model_path = "models/patch_verifier.pt"

    def test_end_to_end_real_frame(self):
        if not os.path.exists(self.model_path):
            self.skipTest("models/patch_verifier.pt not found")
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

    def test_stage1_fallback(self):
        # 400x300 scene
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[:, :] = [70, 40, 15]  # Dark blue water

        # Iceberg
        cv2.circle(img, (260, 140), 50, (220, 225, 225), -1)

        # Boat at (120, 90): red hull with grey cabin
        cv2.rectangle(img, (110, 85), (130, 95), (25, 30, 200), -1)
        cv2.rectangle(img, (116, 88), (124, 92), (150, 150, 150), -1)

        detector = VerifiedDetector(model_path=None, min_color_score=0.25)
        detections = detector.detect(img)
        self.assertGreaterEqual(len(detections), 1, "Should find boat in Stage 1 mode")
        top = detections[0]
        self.assertTrue(top.contains(120, 90, margin=10.0))
        self.assertGreater(top.score, 0.40)


if __name__ == "__main__":
    unittest.main()
