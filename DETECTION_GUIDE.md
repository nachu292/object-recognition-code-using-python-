# Object Recognition Detection Guide

## Quick Start

### For Real-Time Webcam (Fastest)
```bash
python object_recognition.py --fast
```

### For Image Analysis (Balanced)
```bash
python object_recognition.py --source photo.jpg
```

### For Best Quality
```bash
python object_recognition.py --input-size 608
```

---

## Problem: Limited Object Variety Detection

Your YOLOv3-tiny model can detect **80 different object classes**, but may not always detect them all due to:
1. **Detection threshold too high** - Misses lower confidence detections
2. **Object size/clarity** - Very small or blurry objects are harder to detect
3. **Image quality** - Poor lighting, shadows, or occlusion affects detection
4. **Model limitations** - Some object types are naturally harder to detect
5. **Speed vs Accuracy Trade-off** - Faster detection = fewer detections

## Solution: Adjust Detection Parameters

### 1. View All Detectable Objects
```bash
python object_recognition.py --show-all-classes
```
Shows all 80 COCO object classes the model can detect, organized by category:
- PEOPLE (1)
- VEHICLES (8)
- TRAFFIC (4)
- ANIMALS (10)
- CLOTHING & ACCESSORIES (5)
- SPORTS & RECREATION (10)
- KITCHEN & DINING (7)
- FOOD (10)
- FURNITURE & HOUSE (17)
- OTHER (8)

### 2. Speed vs Accuracy Modes

#### ⚡ Fast Mode (Real-Time)
```bash
python object_recognition.py --fast
```
- **Speed:** 50-100ms per frame
- **Input size:** 320×320
- **Threshold:** 0.5
- **Best for:** Live webcam, real-time demos
- **Trade-off:** May miss small objects

#### ⚙️ Balanced Mode (Default)
```bash
python object_recognition.py
# or
python object_recognition.py --source photo.jpg
```
- **Speed:** 100-150ms per frame
- **Input size:** 416×416
- **Threshold:** 0.3
- **Best for:** General use, good balance

#### 🎯 Quality Mode (High Accuracy)
```bash
python object_recognition.py --input-size 608
```
- **Speed:** 200-300ms per frame
- **Input size:** 608×608
- **Best for:** Exhibition quality, detailed analysis
- **Trade-off:** Slower processing

### 3. Detect More Objects with Lower Threshold
```bash
# Use lower threshold to catch weaker detections
python object_recognition.py --source image.jpg --lower-threshold 0.1
```

**How it works:**
- `--lower-threshold 0.1` = Detection threshold (internal processing)
- Default display shows boxes that passed the stricter check

### 4. Typical Usage Examples

#### Basic Detection (Balanced)
```bash
python object_recognition.py --source photo.jpg
# Default: 30% threshold, 416×416 input
```

#### Fast Detection for Live Demo
```bash
python object_recognition.py --fast
# Speed: 50-100ms per frame, real-time capable
```

#### Detect More Objects (Sensitive)
```bash
python object_recognition.py --source photo.jpg --lower-threshold 0.15 --threshold 0.3
# Detects more objects, filters to show only >30% confidence
```

#### High Accuracy (Strict)
```bash
python object_recognition.py --source photo.jpg --min-confidence 0.8
# Only shows very confident detections (>80%)
```

#### Exclude False Positives
```bash
python object_recognition.py --source photo.jpg --exclude "cell phone,remote"
# Detects everything EXCEPT cell phones and remotes
```

#### Fast + No Cell Phones
```bash
python object_recognition.py --fast --no-cell-phone
# Fastest with fewer false positives
```

#### Combined: Sensitive Detection + Exclude Bad Objects
```bash
python object_recognition.py --source photo.jpg --lower-threshold 0.15 --exclude "cell phone,remote,keyboard"
# Detects more objects but hides known false positives
```

### 5. For Video/Webcam

#### Live Webcam (Real-Time)
```bash
python object_recognition.py --fast
# Fast mode with preview
```

#### Webcam Fastest (No Preview Rendering)
```bash
python object_recognition.py --fast --no-preview
# Fastest possible processing
```

#### Video with Detections
```bash
python object_recognition.py --source video.mp4 --save output.mp4
# Default quality

python object_recognition.py --source video.mp4 --save output.mp4 --fast
# Fast processing

python object_recognition.py --source video.mp4 --save output.mp4 --input-size 608
# High quality output
```

#### Exclude Problematic Objects from Video
```bash
python object_recognition.py --source video.mp4 --save output.mp4 --no-cell-phone
```

## Parameter Guide

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `--fast` | OFF | flag | Enable fast mode (320×320 input, faster thresholds) |
| `--gpu` | OFF | flag | Try to use GPU acceleration (if CUDA available) |
| `--threshold` | 0.3 | 0.0-1.0 | Display threshold - minimum confidence to show |
| `--lower-threshold` | 0.1 | 0.0-1.0 | Internal detection threshold - find more objects |
| `--min-confidence` | 0.5 | 0.0-1.0 | Minimum confidence (overrides threshold if higher) |
| `--nms-threshold` | 0.45 | 0.1-0.9 | Box suppression (lower = more overlapping boxes kept) |
| `--exclude` | "" | comma-separated | Exclude specific object types |
| `--input-size` | 416 | 320-640 | YOLO input size (larger = more accurate but slower) |
| `--cell-phone-confidence` | 0.9 | 0.0-1.0 | Stricter filter for cell phone false positives |
| `--no-cell-phone` | OFF | flag | Completely exclude cell phone detections |
| `--max-long-side` | 800 | pixels | Pre-processing image size limit |

## Performance Comparison

| Mode | Input Size | Speed | FPS | Best For |
|------|-----------|-------|-----|----------|
| `--fast` | 320×320 | 50-100ms | 10-20 | Real-time webcam |
| Default | 416×416 | 100-150ms | 7-10 | General use |
| `--input-size 512` | 512×512 | 150-200ms | 5-7 | Good accuracy |
| `--input-size 608` | 608×608 | 200-300ms | 3-5 | High quality |

## What to Try if Objects Still Not Detected

1. **First:** Check available classes
   ```bash
   python object_recognition.py --show-all-classes
   ```

2. **Second:** Lower the thresholds
   ```bash
   python object_recognition.py --source image.jpg --lower-threshold 0.15 --threshold 0.25
   ```

3. **Third:** Increase input size for small objects
   ```bash
   python object_recognition.py --source image.jpg --input-size 512
   ```

4. **Fourth:** Use better quality images
   - Ensure good lighting
   - Objects should be clear and visible
   - Avoid heavy occlusion/overlap

5. **Fifth:** Check if object is in the 80 classes
   ```bash
   python object_recognition.py --show-all-classes
   ```
   If your object isn't listed, it cannot be detected by YOLOv3-tiny.
   ```bash
   python object_recognition.py --source image.jpg --lower-threshold 0.1 --threshold 0.2
   ```

3. **Third:** Increase input size for small objects
   ```bash
   python object_recognition.py --source image.jpg --input-size 512
   ```

4. **Fourth:** Use better quality images
   - Ensure good lighting
   - Objects should be clear and visible
   - Avoid heavy occlusion/overlap

5. **Fifth:** Check if object is in the 80 classes
   ```bash
   python object_recognition.py --show-all-classes
   ```
   If your object isn't listed, it cannot be detected by YOLOv3-tiny.

## Example Scenarios

### Scenario 1: Detect Small Objects
```bash
python object_recognition.py --source cluttered_shelf.jpg --lower-threshold 0.1 --input-size 512
```

### Scenario 2: Detect Objects in Poor Lighting
```bash
python object_recognition.py --source dark_room.jpg --lower-threshold 0.15 --threshold 0.25
```

### Scenario 3: Reduce False Positives
```bash
python object_recognition.py --exclude "cell phone,remote,keyboard" --min-confidence 0.6
```

### Scenario 4: Live Webcam with Good Coverage
```bash
python object_recognition.py --lower-threshold 0.2 --input-size 480
```

## Tips for Best Results

✅ **DO:**
- Use clear, well-lit images
- Have objects clearly visible (not too small, not too blurry)
- Adjust thresholds gradually (try 0.1 → 0.15 → 0.2 → 0.3)
- Exclude known false positives

❌ **DON'T:**
- Use extremely low thresholds (<0.05) - too many false positives
- Try to detect objects not in the 80 COCO classes
- Use very small input sizes with small objects
- Expect 100% accuracy - neural networks have limitations

## Need More Object Classes?

The current YOLOv3-tiny model detects 80 classes from COCO dataset. To detect other object types, you would need to:
1. Use a larger model (YOLOv3 Full - slower but more accurate)
2. Use a different model (YOLOv5, YOLOv8)
3. Fine-tune the model on custom objects (requires training)

However, this requires installing additional packages which you've chosen not to do currently.
