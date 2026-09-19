"""Tests for Step 1: Color-anomaly candidate detector."""
import unittest
import numpy as np
import cv2

from tools.detect_color import ColorAnomalyDetector


class TestColorAnomalyDetector(unittest.TestCase):
    def setUp(self):
        self.detector = ColorAnomalyDetector(min_area=2, min_score=0.30, patch_size=48)

    def test_synthetic_arctic_scene(self):
        # Create a 400x300 synthetic Arctic scene:
        # Background: Dark blue water (BGR: [70, 40, 15])
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[:, :] = [70, 40, 15]

        # Add an ice floe: Grey-white (BGR: [215, 220, 220])
        cv2.circle(img, (250, 150), 60, (215, 220, 220), -1)

        # Target 1: Clearly visible red boat (12x8 pixels) at (100, 80)
        # BGR: [20, 25, 195]
        img[76:84, 94:106] = [20, 25, 195]

        # Target 2: Distant small boat (4x3 pixels, subpixel blended pinkish) at (320, 220)
        # BGR: [65, 45, 160]
        img[219:222, 318:322] = [65, 45, 160]

        candidates = self.detector.detect(img)
        self.assertGreaterEqual(len(candidates), 1, "Should find at least 1 boat candidate")

        # Top candidate should be the main red boat around (100, 80)
        c0 = candidates[0]
        self.assertTrue(c0.contains(100, 80, margin=5.0),
                        f"Expected candidate near (100, 80), got ({c0.cx}, {c0.cy})")
        self.assertGreater(c0.score, 0.5, "Top candidate should have high confidence")

        # Check patch extraction
        self.assertIsNotNone(c0.patch)
        self.assertEqual(c0.patch.shape, (48, 48, 3))


if __name__ == "__main__":
    unittest.main()
