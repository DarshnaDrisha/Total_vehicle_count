# Vehicle Counting Submission

This project contains a submission-ready Python implementation for **vehicle counting from a video**.

The code follows the required submission format:
- A class named **`Solution`**
- A method named **`forward(video_path)`**
- `forward()` accepts a **video file path** and returns the **total vehicle count (int)**

---

## Setup

### 1) Create environment (recommended)
```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
```

### 2) Install dependencies
```bash
pip install opencv-python numpy
```

---

## Execution / Usage

### Use as Python module (for evaluation)
```python
from src.solution import Solution

sol = Solution()
count = sol.forward("/path/to/input_video.mp4")
print(count)   # integer total vehicle count
```

### Optional: run directly for local testing
```bash
python src/solution.py /path/to/input_video.mp4
```

---

## Processing pipeline — concise explanation

The algorithm counts vehicles using **background subtraction + motion-based blob detection** and a **crossing line/strip** logic.

### 1. Video decoding & preprocessing
- Read frames from the input video using OpenCV.
- Resize frames to a fixed standard size for stable processing speed and thresholds.
- Convert frames to grayscale.

### 2. Median background estimation
- Uniformly sample frames from across the video.
- Compute the **per-pixel median** of sampled grayscale frames.
- This median frame represents a robust background image (less sensitive to moving vehicles).

### 3. Foreground (motion) mask generation
For each frame:
- Compute absolute difference between current grayscale frame and the median background.
- Apply thresholding to obtain a binary motion mask.
- Apply morphological operations (opening + closing) to remove noise and fill gaps.

### 4. Road ROI estimation
- Build an accumulated motion map over sampled frames.
- Extract the dominant active region and compute a bounding box ROI.
- This ROI restricts processing to road area and reduces false positives.

### 5. Motion direction estimation
- Optical flow is computed on sample frame pairs inside ROI.
- Average flow direction determines if vehicles mainly move **downwards** or **upwards**.

### 6. Blob extraction and filtering
- Run connected components on the ROI foreground mask.
- Filter blobs using:
  - minimum/maximum area
  - aspect ratio constraints
- Refine overlapping blobs to avoid duplicate detections.

### 7. Counting with decision strip
- A horizontal **decision strip** is defined inside ROI.
- A vehicle is counted when:
  - its centroid enters the strip, and
  - it was observed previously on the opposite side (based on centroid history / fg history)
- A recent history buffer prevents double counting.

### 8. Output
- The final count is returned as:
  ```python
  return int(total_count)
  ```

---

## Output
The output of `forward(video_path)` is:
- **Type:** `int`
- **Meaning:** total number of vehicles counted in the video

---

## Notes
- Works best with a fixed/static camera.
- If you see undercounting/overcounting, tune thresholds such as:
  - `DIFF_THRESH`
  - `MIN_BLOB_AREA`, `MAX_BLOB_AREA`
  - `STRIP_REL_POS`, `STRIP_WIDTH_PX`
