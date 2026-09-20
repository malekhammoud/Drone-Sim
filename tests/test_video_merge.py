"""Unit tests for tools/video_merge.py."""
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from tools.video_merge import merge_videos_top_down


class TestVideoMerge(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.top_mp4 = os.path.join(self.tmpdir, "top.mp4")
        self.bottom_mp4 = os.path.join(self.tmpdir, "bottom.mp4")
        self.out_mp4 = os.path.join(self.tmpdir, "merged.mp4")

        # Create two small test videos
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Top: 1280x720, 5 frames
        w_top = cv2.VideoWriter(self.top_mp4, fourcc, 5.0, (1280, 720))
        for i in range(5):
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            img[:] = (50 * i, 100, 150)
            w_top.write(img)
        w_top.release()

        # Bottom: 960x720, 4 frames (like quad camera)
        w_bot = cv2.VideoWriter(self.bottom_mp4, fourcc, 4.0, (960, 720))
        for i in range(4):
            img = np.zeros((720, 960, 3), dtype=np.uint8)
            img[:] = (100, 50 * i, 200)
            w_bot.write(img)
        w_bot.release()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_merge_parallel(self):
        out = merge_videos_top_down(
            self.top_mp4, self.bottom_mp4, self.out_mp4,
            target_width=1280, pane_height=720, mode="parallel"
        )
        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 0)

        cap = cv2.VideoCapture(out)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.assertEqual(w, 1280)
        self.assertEqual(h, 720 * 2 + 44)
        self.assertEqual(count, 5)  # max(5, 4)

    def test_merge_sequential(self):
        seq_out = os.path.join(self.tmpdir, "seq_merged.mp4")
        out = merge_videos_top_down(
            self.top_mp4, self.bottom_mp4, seq_out,
            target_width=1280, pane_height=720, mode="sequential"
        )
        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 0)

        cap = cv2.VideoCapture(out)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.assertEqual(w, 1280)
        self.assertEqual(h, 720 * 2 + 44)
        self.assertEqual(count, 9)  # 5 + 4


if __name__ == "__main__":
    unittest.main()
