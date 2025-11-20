"""
Lightweight object recognition demo that avoids heavyweight ML runtimes.

This version uses OpenCV's DNN module with YOLOv3-tiny Darknet weights
(downloaded from pjreddie.com and mirrors). Benefits:
  * Runs anywhere OpenCV does and needs only `opencv-python` + `numpy`.
  * Automatically downloads the pretrained weights/config the first time it runs.

Usage examples:
  python object_recognition.py                 # Live webcam preview
  python object_recognition.py --source img.jpg --save result.jpg
  python object_recognition.py --source video.mp4 --save annotated.mp4
"""

from __future__ import annotations

import argparse
import pathlib
import time
import urllib.error
import urllib.request
from typing import List, Tuple

import cv2
import numpy as np


COCO_LABELS: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Extended object categories for better identification
EXTENDED_LABELS: List[str] = [
    "person", "pedestrian", "cyclist", "child", "adult",
    "vehicle", "car", "truck", "bus", "motorcycle", "bicycle", "scooter", "skateboard",
    "road", "sidewalk", "building", "house", "store", "tree", "grass", "sky", "water",
    "animal", "dog", "cat", "bird", "horse", "cow", "sheep", "zebra", "elephant", "bear", "giraffe",
    "furniture", "chair", "table", "bed", "sofa", "couch", "desk", "cabinet", "shelf",
    "food", "apple", "banana", "orange", "pizza", "burger", "sandwich", "cake", "donut",
    "drink", "bottle", "cup", "glass", "wine glass", "beer",
    "sport", "ball", "bat", "racket", "skateboard", "surfboard", "skis", "snowboard",
    "electronics", "laptop", "phone", "mouse", "keyboard", "remote", "tv", "monitor",
    "kitchen", "microwave", "oven", "toaster", "sink", "refrigerator", "stove",
    "outdoor", "traffic light", "stop sign", "bench", "parking meter", "hydrant",
    "misc", "umbrella", "backpack", "handbag", "suitcase", "tie", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

CFG_URLS = [
    "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
    "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg",
]
WEIGHTS_URLS = [
    "https://pjreddie.com/media/files/yolov3-tiny.weights",
    "https://archive.org/download/yolov3-tiny/yolov3-tiny.weights",
]
MODEL_DIR = pathlib.Path("models")
CFG_FILE = "yolov3-tiny.cfg"
WEIGHTS_FILE = "yolov3-tiny.weights"
DEFAULT_INPUT_SIZE = 416


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime object recognition with a lightweight detector."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="Path to an image/video file or 'webcam' (default).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Path to save annotated output (image or video).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Score threshold for keeping detections.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.45,
        help="Non-maximum suppression threshold (lower keeps more boxes).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=416,
        help="Square input dimension for YOLO (multiples of 32, e.g., 416, 512, 608).",
    )
    parser.add_argument(
        "--max-long-side",
        type=int,
        default=800,
        help="Resize longest image side to this many pixels before inference.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help="Preferred webcam index (default: auto-detect among 0-3).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip on-screen preview (useful for headless runs).",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated list of object names to exclude from detection (e.g., 'cell phone,remote').",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence score for an object (0.0-1.0). Higher = stricter, fewer false detections.",
    )
    parser.add_argument(
        "--lower-threshold",
        type=float,
        default=0.1,
        help="Lower threshold for detecting smaller/faint objects (0.0-1.0). Use with caution.",
    )
    parser.add_argument(
        "--show-all-classes",
        action="store_true",
        help="Show all 80 COCO classes that the model can detect.",
    )
    parser.add_argument(
        "--no-cell-phone",
        action="store_true",
        help="Automatically exclude cell phone detections (common false positive).",
    )
    parser.add_argument(
        "--cell-phone-confidence",
        type=float,
        default=0.9,
        help="Minimum confidence for cell phone detection (0.0-1.0). Higher = stricter. Default: 0.9",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode: smaller input size (320) and higher thresholds for speed.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Try to use GPU acceleration (requires CUDA).",
    )
    return parser.parse_args()


def show_available_classes() -> None:
    """Display all 80 COCO object classes the model can detect."""
    print("\n" + "="*60)
    print("YOLO v3-tiny can detect the following 80 object classes:")
    print("="*60)

    # Group by category for easier reading
    categories = {
        "PEOPLE": ["person"],
        "VEHICLES": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
        "TRAFFIC": ["traffic light", "fire hydrant", "stop sign", "parking meter"],
        "ANIMALS": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
        "CLOTHING & ACCESSORIES": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
        "SPORTS & RECREATION": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"],
        "KITCHEN & DINING": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
        "FOOD": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
        "FURNITURE & HOUSE": ["chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator"],
        "OTHER": ["bench", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
    }

    for category, objects in categories.items():
        print(f"\n{category}:")
        for i, obj in enumerate(objects, 1):
            print(f"  {i:2d}. {obj}")

    print("\n" + "="*60)
    print("Total: 80 object classes")
    print("="*60 + "\n")


def _download_with_mirrors(urls: List[str], destination: pathlib.Path, label: str) -> None:
    for url in urls:
        try:
            print(f"Downloading {label} from {url} …")
            urllib.request.urlretrieve(url, destination)
            return
        except urllib.error.URLError as err:
            print(f"  Failed from {url}: {err}. Trying next mirror…")
    raise RuntimeError(
        f"Unable to download {label}. Please download it manually and place it at {destination}."
    )


def ensure_model_files() -> Tuple[pathlib.Path, pathlib.Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = MODEL_DIR / CFG_FILE
    weights_path = MODEL_DIR / WEIGHTS_FILE
    if not cfg_path.exists():
        _download_with_mirrors(CFG_URLS, cfg_path, CFG_FILE)
    if not weights_path.exists():
        _download_with_mirrors(WEIGHTS_URLS, weights_path, WEIGHTS_FILE)
    return cfg_path, weights_path


def load_model(input_size: int = DEFAULT_INPUT_SIZE, use_gpu: bool = False) -> cv2.dnn_DetectionModel:
    cfg_path, weights_path = ensure_model_files()
    model = cv2.dnn.DetectionModel(str(cfg_path), str(weights_path))
    model.setInputSize(input_size, input_size)
    model.setInputScale(1 / 255.0)
    model.setInputSwapRB(True)

    # Try to use GPU if requested
    if use_gpu:
        try:
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("GPU acceleration enabled (CUDA).")
        except Exception as e:
            print(f"GPU acceleration not available: {e}. Using CPU.")
    else:
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return model


def preprocess(frame: np.ndarray, max_long_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, max_long_side / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    return frame


def run_detection(
    model: cv2.dnn_DetectionModel,
    frame: np.ndarray,
    threshold: float,
    nms_threshold: float,
    max_long_side: int,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    processed = preprocess(frame, max_long_side)
    class_ids, confidences, boxes = model.detect(
        processed, confThreshold=threshold, nmsThreshold=nms_threshold
    )
    if class_ids is None:
        detections = (np.array([]), np.array([]), np.array([]))
    else:
        class_ids = np.array(class_ids).reshape(-1)
        confidences = np.array(confidences).reshape(-1)
        boxes = np.array(boxes).reshape(-1, 4)
        detections = (class_ids, confidences, boxes)
    return processed, detections


def get_object_stats(
    detections: Tuple[np.ndarray, np.ndarray, np.ndarray],
    threshold: float,
    exclude_objects: set = None,
    cell_phone_confidence: float = 0.9,
) -> dict:
    """Count and categorize detected objects, with smart cell phone filtering."""
    if exclude_objects is None:
        exclude_objects = set()

    class_ids, confidences, boxes = detections
    stats = {}
    cell_phone_id = 61  # Index of 'cell phone' in COCO_LABELS

    for label, score in zip(class_ids, confidences):
        if score < threshold:
            continue
        name = COCO_LABELS[int(label)] if int(
            label) < len(COCO_LABELS) else "unknown object"

        # Skip excluded objects
        if name.lower() in exclude_objects:
            continue

        # Apply stricter filter for cell phone false positives
        if int(label) == cell_phone_id and score < cell_phone_confidence:
            continue

        stats[name] = stats.get(name, 0) + 1
    return stats


def annotate(
    frame,
    detections: Tuple[np.ndarray, np.ndarray, np.ndarray],
    threshold: float,
    color: Tuple[int, int, int] = (30, 220, 30),
    show_confidence: bool = True,
    exclude_objects: set = None,
    cell_phone_confidence: float = 0.9,
) -> None:
    if exclude_objects is None:
        exclude_objects = set()

    class_ids, confidences, boxes = detections
    cell_phone_id = 61  # Index of 'cell phone' in COCO_LABELS

    for label, score, (x, y, w, h) in zip(class_ids, confidences, boxes):
        if score < threshold:
            continue
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        name = COCO_LABELS[int(label)] if int(
            label) < len(COCO_LABELS) else "object"

        # Skip excluded objects
        if name.lower() in exclude_objects:
            continue

        # Apply stricter filter for cell phone false positives
        if int(label) == cell_phone_id and score < cell_phone_confidence:
            continue

        caption = f"{name} {score:.2f}" if show_confidence else name
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            caption,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def image_mode(
    model: cv2.dnn_DetectionModel,
    source_path: pathlib.Path,
    threshold: float,
    nms_threshold: float,
    max_long_side: int,
    exclude_objects: set = None,
    cell_phone_confidence: float = 0.9,
):
    if exclude_objects is None:
        exclude_objects = set()

    frame = cv2.imread(str(source_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {source_path}")
    resized, detections = run_detection(
        model, frame, threshold, nms_threshold, max_long_side)
    annotate(resized, detections, threshold, exclude_objects=exclude_objects,
             cell_phone_confidence=cell_phone_confidence)

    # Get object statistics
    stats = get_object_stats(detections, threshold,
                             exclude_objects=exclude_objects, cell_phone_confidence=cell_phone_confidence)
    return resized, detections, stats


def video_mode(
    model: cv2.dnn_DetectionModel,
    source: cv2.VideoCapture,
    threshold: float,
    nms_threshold: float,
    max_long_side: int,
    preview: bool,
    save_path: pathlib.Path | None,
    fps: float,
    exclude_objects: set = None,
    cell_phone_confidence: float = 0.9,
) -> None:
    if exclude_objects is None:
        exclude_objects = set()

    prev = time.perf_counter()
    writer = None
    while True:
        ok, frame = source.read()
        if not ok:
            break
        resized, detections = run_detection(
            model, frame, threshold, nms_threshold, max_long_side)
        annotate(resized, detections, threshold,
                 exclude_objects=exclude_objects, cell_phone_confidence=cell_phone_confidence)

        now = time.perf_counter()
        fps_display = 1.0 / max(1e-6, now - prev)
        prev = now
        cv2.putText(
            resized,
            f"{fps_display:.1f} FPS",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 180, 30),
            2,
            cv2.LINE_AA,
        )

        if preview:
            cv2.imshow("Object Recognition", resized)
            # Exit on 'q' press or if the window is closed.
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q") or cv2.getWindowProperty("Object Recognition", cv2.WND_PROP_VISIBLE) < 1:
                break
        if save_path and writer is None:
            writer = build_writer(
                save_path, (resized.shape[1], resized.shape[0]), fps)
        if writer is not None:
            writer.write(resized)

    source.release()
    if writer is not None:
        writer.release()
    if preview:
        cv2.destroyAllWindows()


def build_writer(save_path: pathlib.Path, frame_size: Tuple[int, int], fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {save_path}")
    return writer


def open_webcam(preferred_index: int) -> cv2.VideoCapture:
    indices = [preferred_index] if preferred_index >= 0 else list(range(6))
    backends = [None, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    errors = []
    for index in indices:
        for backend in backends:
            if backend is None:
                cap = cv2.VideoCapture(index)
                backend_name = "DEFAULT"
            else:
                cap = cv2.VideoCapture(index, backend)
                backend_name = str(backend)
            if cap.isOpened():
                print(
                    f"Using webcam index {index} with backend {backend_name}")
                return cap
            errors.append(f"index {index} backend {backend_name}")
            cap.release()
    raise RuntimeError(
        "Could not access webcam after trying "
        + ", ".join(errors)
        + ". If you do not have a camera, pass --source path/to/video or image."
    )


def main() -> None:
    args = parse_args()

    # Show available classes if requested
    if args.show_all_classes:
        show_available_classes()
        return

    # Handle fast mode
    input_size = args.input_size
    threshold = args.threshold
    nms_threshold = args.nms_threshold
    max_long_side = args.max_long_side

    if args.fast:
        print("⚡ Fast mode enabled")
        input_size = 320  # Smaller input = faster
        threshold = 0.5   # Higher threshold = fewer detections but faster
        nms_threshold = 0.5
        max_long_side = 480  # Smaller preprocessing
        print(f"  Input size: {input_size}px")
        print(f"  Threshold: {threshold}")
        print(f"  NMS threshold: {nms_threshold}")

    model = load_model(input_size, use_gpu=args.gpu)
    source_is_webcam = args.source.lower() == "webcam"
    save_path = pathlib.Path(args.save) if args.save else None

    # Parse excluded objects
    exclude_objects = set(obj.strip().lower()
                          for obj in args.exclude.split(",") if obj.strip())

    # Auto-exclude cell phone if --no-cell-phone flag is set
    if args.no_cell_phone:
        exclude_objects.add("cell phone")
        print("Excluding cell phone detections (--no-cell-phone flag set).")

    # Determine cell phone confidence threshold
    cell_phone_confidence = args.cell_phone_confidence
    if cell_phone_confidence < 1.0:
        print(
            f"Using stricter cell phone filter (confidence ≥ {cell_phone_confidence:.2f}).")

    # Use min_confidence if it's higher than threshold
    threshold = max(threshold, args.min_confidence)

    # Use lower_threshold for detection if specified and lower than threshold
    if args.lower_threshold < threshold:
        detection_threshold = args.lower_threshold
        print(
            f"Using lower detection threshold ({detection_threshold:.2f}) to detect more objects.")
        print(f"Display threshold: {threshold:.2f}")
    else:
        detection_threshold = threshold

    if threshold != args.threshold and not args.fast:
        print(
            f"Using higher threshold ({threshold:.2f}) from --min-confidence parameter.")

    if source_is_webcam:
        cap = open_webcam(args.camera_index)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        video_mode(
            model,
            cap,
            detection_threshold,
            nms_threshold,
            max_long_side,
            not args.no_preview,
            save_path,
            fps,
            exclude_objects=exclude_objects,
            cell_phone_confidence=cell_phone_confidence,
        )
    else:
        source_path = pathlib.Path(args.source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")
        if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            annotated, detections, stats = image_mode(
                model,
                source_path,
                detection_threshold,
                nms_threshold,
                max_long_side,
                exclude_objects=exclude_objects,
                cell_phone_confidence=cell_phone_confidence,
            )
            if save_path:
                cv2.imwrite(str(save_path), annotated)
                print(f"Saved annotated image to {save_path}")
            if not args.no_preview:
                cv2.imshow("Object Recognition", annotated)
                while cv2.getWindowProperty("Object Recognition", cv2.WND_PROP_VISIBLE) >= 1:
                    if cv2.waitKey(100) != -1:
                        break
                cv2.destroyAllWindows()
            kept = len(detections[0])
            print(f"\nDetected {kept} objects (threshold={threshold:.2f}).")
            if exclude_objects:
                print(
                    f"Excluded objects: {', '.join(sorted(exclude_objects))}")
            if stats:
                print("\nObject breakdown:")
                for obj_name in sorted(stats.keys()):
                    print(f"  {obj_name}: {stats[obj_name]}")
            else:
                print("No objects detected. Try lowering the threshold.")
        else:
            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {source_path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            video_mode(
                model,
                cap,
                detection_threshold,
                nms_threshold,
                max_long_side,
                not args.no_preview,
                save_path,
                fps,
                exclude_objects=exclude_objects,
                cell_phone_confidence=cell_phone_confidence,
            )


if __name__ == "__main__":
    main()
