# Project Dependencies

This project uses OpenCV's DNN module with pretrained YOLOv3-tiny weights and a small Streamlit front-end. Below are the Python libraries needed to run the scripts in this repository, plus optional packages and notes for GPU acceleration and headless setups.

## Required Python packages

- `opencv-python` — OpenCV bindings (DNN backend used for detection)
- `numpy` — Numeric arrays and image handling

If you plan to use the Streamlit front-end (`streamlit_app.py`):
- `streamlit` — Web UI for uploading / previewing images
- `pillow` (PIL) — Image handling for Streamlit

## Optional / useful packages

- `opencv-contrib-python` — OpenCV with extra modules (if you need contrib features)
- `opencv-python-headless` — Headless OpenCV (no GUI) for servers; do NOT install this if you need `cv2.imshow` on desktop
- `matplotlib` — Helpful for plotting debug images (optional)

## Installation (recommended)

Create and activate a virtual environment first (Windows PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install packages with pip:

```powershell
pip install --upgrade pip
pip install opencv-python numpy
```

If you want the Streamlit UI:

```powershell
pip install streamlit pillow
```

Or install everything at once:

```powershell
pip install opencv-python numpy streamlit pillow
```

## Using a requirements file

You can create a `requirements.txt` with the following contents and install it:

```
opencv-python
numpy
streamlit
pillow
```

Then run:

```powershell
pip install -r requirements.txt
```

## Notes about GPU / CUDA acceleration

- The `--gpu` option in `object_recognition.py` attempts to configure OpenCV's DNN backend to use CUDA.
- The PyPI `opencv-python` wheel is built for CPU only and does not include CUDA.
- To get GPU acceleration you must use an OpenCV build compiled with CUDA support. Options:
  - Build OpenCV from source with CUDA support (advanced).
  - Use a prebuilt OpenCV wheel that includes CUDA (rare; check trusted sources for your platform).
  - Use a Docker image with OpenCV+CUDA already configured.

If GPU is not available or OpenCV lacks CUDA support, the script will fall back to CPU automatically.

## Headless / Server usage

- If running on a headless server (no display), replace `opencv-python` with `opencv-python-headless` and do not use functions that open a GUI window (`cv2.imshow`).

```powershell
pip uninstall opencv-python
pip install opencv-python-headless
```

- For Streamlit deployments, you still need `streamlit` and `pillow`.

## Python version

- This project runs on Python 3.8+ (the user environment shows Python 3.12 available). Use a modern 3.x interpreter.

## Model files

- The YOLO config and weights (`yolov3-tiny.cfg`, `yolov3-tiny.weights`) are downloaded automatically by the script to the `models/` directory on first run. No manual package required for these files.

## Troubleshooting

- If import fails for `cv2`, reinstall OpenCV:

```powershell
pip install --force-reinstall opencv-python
```

- If `streamlit` commands fail, upgrade pip and reinstall Streamlit:

```powershell
pip install --upgrade pip
pip install --force-reinstall streamlit
```

- If you need GPU acceleration but can't build OpenCV, consider running on CPU with `--fast`, reduce input size, or use cloud/VM instances with prebuilt OpenCV+CUDA images.

---

If you want, I can also create a `requirements.txt` file in the repo and/or add exact pinned versions (e.g. `opencv-python==4.7.0.72`, `numpy==1.25.0`) for reproducibility. Which would you prefer?