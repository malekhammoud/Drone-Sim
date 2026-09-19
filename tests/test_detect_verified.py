"""Tests for Step 2: Verified Detector (Step 1 + Step 2 pipeline)."""
import unittest
import numpy as np
import cv2
import os

from tools.detect_verified import VerifiedDetector


class TestVerifiedDetector(unittest.TestCase):
    def setUp(self):
        model_path = "models/patch_verifier.pt"
        if not os.path.exists(model_path):
            self.skipTest("models/patch_verifier.pt not found")
        self.detector = VerifiedDetector(model_path=model_path, min_color_score=0.25, min_verify_prob=0.30)

    def test_end_to_end_synthetic(self):
        # 400x300 scene
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[:, :] = [70, 40, 15]  # Dark blue water

        # Iceberg
        cv2.circle(img, (260, 140), 50, (220, 225, 225), -1)

        # Boat at (120, 90): red hull with grey cabin
        cv2.rectangle(img, (110, 85), (130, 95), (25, 30, 200), -1)
        cv2.rectangle(img, (116, 88), (124, 92), (150, 150, 150), -1)

        detections = self.detector.detect(img)
        self.assertGreaterEqual(len(detections), 1, "Should find verified boat")

        top = detections[0]
        self.assertTrue(top.contains(120, 90, margin=10.0))
        self.assertGreater(top.score, 0.40)


if __name__ == "__main__":
    unittest.main()
