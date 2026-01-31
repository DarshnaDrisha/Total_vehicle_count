#!/usr/bin/env python3
"""
Submission-ready wrapper around vehicle_count_workswell logic.

Contains:
- class Solution with method forward(video_path) -> int

The forward method runs the counting pipeline on the provided video path
and returns the total vehicle count as an integer.
"""

import cv2
import numpy as np
import os
from collections import deque

# Tunable defaults (kept from original script)
STANDARD_W = 1312
STANDARD_H = 1074
MAX_BG_SAMPLES = 500
BG_BLUR = 3
DIFF_THRESH = 50
MORPH_OPEN = (3, 3)
MORPH_CLOSE = (9, 9)
MIN_BLOB_AREA = 1000
MAX_BLOB_AREA = 600000
ASPECT_MIN = 0.25
ASPECT_MAX = 4.0
STRIP_REL_POS = 0.2
STRIP_WIDTH_PX = 70
NEARBY_DIST_PX = 200


def refine_bboxes(detections):
    def overlaps(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    detections = sorted(detections, key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True)
    kept = []
    for det in detections:
        box = det["bbox"]
        if all(not overlaps(box, k["bbox"]) for k in kept):
            kept.append(det)
    return kept


def sample_frame_indices(total_frames, max_samples):
    if total_frames <= max_samples:
        return list(range(total_frames))
    step = total_frames / float(max_samples)
    return [int(i * step) for i in range(max_samples)]


def compute_median_background(cap, sample_idx):
    frames = []
    # helper to pick odd blur kernel
    def odd_kernel(k):
        return k if (k % 2 == 1 and k > 0) else (k + 1)

    for i in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret or f is None:
            continue
        f = cv2.resize(f, (STANDARD_W, STANDARD_H))
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if BG_BLUR and BG_BLUR > 0:
            k = odd_kernel(BG_BLUR)
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        frames.append(gray)
    if len(frames) == 0:
        raise RuntimeError("No frames sampled for background")
    stack = np.stack(frames, axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)
    return median


def fg_mask_from_frame(frame_gray, bg_median):
    diff = cv2.absdiff(frame_gray, bg_median)
    _, th = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_OPEN)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_CLOSE)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close)
    return th


def is_near_recent_counts(pt, recent, current_frame_idx, dist_thresh=NEARBY_DIST_PX):
    for (cx, cy, frame_idx) in recent:
        if abs(frame_idx - current_frame_idx) < 20:
            if (abs(cx - pt[0]) <= dist_thresh and abs(cy - pt[1]) <= dist_thresh):
                return True
    return False


class Solution:
    """
    Submission class. Use forward(video_path) -> int to get total vehicle count.
    """

    def forward(self, video_path: str) -> int:
        """
        Run the vehicle counting pipeline on video_path and return the total count.

        :param video_path: Path to input video file
        :return: total vehicle count (int)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = STANDARD_W
        H = STANDARD_H

        # guard: if reported total_frames is zero, attempt to read at least once to estimate
        if total_frames <= 0:
            # try counting manually (not efficient but safe)
            tmp_count = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                tmp_count += 1
            total_frames = tmp_count
            cap.release()
            cap = cv2.VideoCapture(video_path)

        # compute median background from sampled frames
        sample_idx = sample_frame_indices(total_frames, min(MAX_BG_SAMPLES, max(1, total_frames)))
        bg_median = compute_median_background(cap, sample_idx)

        # build motion-based ROI using the sampled indices
        accum = np.zeros_like(bg_median, dtype=np.uint32)
        for idx in sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret or f is None:
                continue
            f = cv2.resize(f, (STANDARD_W, STANDARD_H))
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            d = cv2.absdiff(gray, bg_median)
            _, t = cv2.threshold(d, DIFF_THRESH, 255, cv2.THRESH_BINARY)
            accum += t.astype(np.uint32)
        norm = (accum / max(1, len(sample_idx))).astype(np.uint8)
        _, binm = cv2.threshold(norm, 20, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            mask = np.zeros_like(binm)
            cv2.drawContours(mask, cnts, -1, 255, -1)
            x, y, w, h = cv2.boundingRect(mask)
            padx = int(0.03 * W)
            pady = int(0.03 * H)
            x1 = max(0, x - padx)
            y1 = max(0, y - pady)
            w1 = min(W - x1, w + 2 * padx)
            h1 = min(H - y1, h + 2 * pady)
            road_roi = (x1, y1, w1, h1)
        else:
            road_roi = (0, int(H * 0.35), W, int(H * 0.6))

        rx, ry, rw, rh = road_roi

        # estimate direction of motion (downwards True, upwards False)
        def estimate_direction(cap_local, samplecnt=6):
            votes = []
            for i in range(0, samplecnt):
                f1_idx = int(i * max(1, total_frames // (samplecnt + 1)))
                f2_idx = min(total_frames - 1, f1_idx + 3)
                cap_local.set(cv2.CAP_PROP_POS_FRAMES, f1_idx)
                ret1, fr1 = cap_local.read()
                cap_local.set(cv2.CAP_PROP_POS_FRAMES, f2_idx)
                ret2, fr2 = cap_local.read()
                if not (ret1 and ret2 and fr1 is not None and fr2 is not None):
                    continue
                fr1 = cv2.resize(fr1, (STANDARD_W, STANDARD_H))
                fr2 = cv2.resize(fr2, (STANDARD_W, STANDARD_H))
                g1 = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)[ry:ry + rh, rx:rx + rw]
                g2 = cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY)[ry:ry + rh, rx:rx + rw]
                # Farneback optical flow
                flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mean_dy = np.mean(flow[..., 1])
                votes.append(mean_dy)
            if not votes:
                return True
            avg = float(np.mean(votes))
            return avg > 0

        direction_down = estimate_direction(cap)

        # Decision strip
        strip_center_y = int(ry + rh * STRIP_REL_POS)
        strip_y1 = max(0, strip_center_y - STRIP_WIDTH_PX // 2)
        strip_y2 = min(H, strip_center_y + STRIP_WIDTH_PX // 2)

        # reset capture for main loop
        cap.release()
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        total_count = 0
        recent_counts = deque(maxlen=500)

        # temporal buffers
        LOOK_BACK = 6
        prev_fg = deque(maxlen=LOOK_BACK)
        prev_centroids = deque(maxlen=LOOK_BACK)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1
            frame = cv2.resize(frame, (STANDARD_W, STANDARD_H))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg = fg_mask_from_frame(gray, bg_median)

            roi_mask = fg[ry:ry + rh, rx:rx + rw].copy()
            num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(roi_mask, connectivity=8)
            detections = []
            for i in range(1, num_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
                    continue
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                ar = w / float(h) if h > 0 else 0
                if ar < ASPECT_MIN or ar > ASPECT_MAX:
                    continue
                cx_roi, cy_roi = cents[i]
                cx_full = int(cx_roi + rx)
                cy_full = int(cy_roi + ry)
                x_full = int(stats[i, cv2.CC_STAT_LEFT] + rx)
                y_full = int(stats[i, cv2.CC_STAT_TOP] + ry)
                detections.append({
                    "centroid": (cx_full, cy_full),
                    "bbox": (x_full, y_full, int(w), int(h)),
                    "area": area
                })

            detections = refine_bboxes(detections)

            # counting logic
            for det in detections:
                cx, cy = det["centroid"]
                inside_strip_now = (strip_y1 <= cy <= strip_y2)
                if not inside_strip_now:
                    continue

                was_on_far_side = False

                # centroid matching against history
                for k in range(1, len(prev_centroids) + 1):
                    prev_list = prev_centroids[-k]
                    for (pcx, pcy) in prev_list:
                        if abs(pcx - cx) <= NEARBY_DIST_PX and abs(pcy - cy) <= NEARBY_DIST_PX:
                            was_on_far_side = True
                            break
                    if was_on_far_side:
                        break

                # fg fallback check
                if not was_on_far_side:
                    if direction_down:
                        sample_y = int(np.clip(cy - (STRIP_WIDTH_PX + 8), 0, H - 1))
                    else:
                        sample_y = int(np.clip(cy + (STRIP_WIDTH_PX + 8), 0, H - 1))
                    sample_x = int(np.clip(cx, 0, W - 1))
                    for k in range(1, len(prev_fg) + 1):
                        if prev_fg[-k][sample_y, sample_x] > 0:
                            was_on_far_side = True
                            break

                if was_on_far_side:
                    if not is_near_recent_counts((cx, cy), recent_counts, frame_idx):
                        total_count += 1
                        recent_counts.append((cx, cy, frame_idx))

            # append buffers for next frames
            prev_fg.append(fg.copy())
            prev_centroids.append([d["centroid"] for d in detections])

        cap.release()

        # final result: ensure int
        return int(total_count)


# If run as script, provide a minimal CLI for testing (not required for submission).
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python submission.py <video_path>")
        sys.exit(1)
    video = sys.argv[1]
    sol = Solution()
    count = sol.forward(video)
    print("Total vehicle count:", count)

