# Models folder — README

This folder contains model configuration/weight filenames used by the project, but the large binary weight file is not committed to the repository.

Why the weight file is not included in the repo
- The pretrained YOLO weight file (`yolov3-tiny.weights`) is a large binary (tens of megabytes). Many Git repositories avoid committing large binaries to keep the repository small, and some hosting platforms (or team policies) prohibit storing large files directly in the git history.
- The code in this project downloads the required files automatically on first run when an internet connection is available.

Files referenced by the project
- `yolov3-tiny.cfg` — network configuration (text). This file may be included in the repository.
- `yolov3-tiny.weights` — pretrained model weights (binary). **This file is not included** due to size.

What to do if the files are missing
1. If you have an internet connection, run the script; it will attempt to download the missing files automatically into this folder. Example:

   ```powershell
   python ..\object_recognition.py --show-all-classes
   ```

   On first run the script prints download attempts and will save the files under `models/`.

2. If your machine has no internet access, download the required files on another machine and copy them here (USB or network share). Filenames must match exactly:

   - `yolov3-tiny.cfg`
   - `yolov3-tiny.weights`

Recommended official download sources (mirrors used by the script)
- cfg (text):
  - https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
- weights (binary):
  - https://pjreddie.com/media/files/yolov3-tiny.weights

PowerShell download example (run from the project root)

```powershell
# create models folder if needed
mkdir .\models

# download cfg
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg" -OutFile .\models\yolov3-tiny.cfg

# download weights (large file)
Invoke-WebRequest -Uri "https://pjreddie.com/media/files/yolov3-tiny.weights" -OutFile .\models\yolov3-tiny.weights
```

Notes and caveats
- The weights file is a binary file — do not attempt to open it in a text editor.
- If your network blocks direct downloads, use a machine with internet to fetch the files and copy them into this folder.
- If you prefer an automated script, create `download_models.ps1` with the commands above and run it.

AI prompt you can paste to an assistant to fetch/create the missing files

> IMPORTANT: Most AI models cannot fabricate pretrained binary model weights from scratch — they can only provide instructions or scripts to download them from the official sources. Use the prompt below to request an AI helper to create a downloadable script or to fetch the files from the official sources and place them into the `models/` directory.

AI prompt (copy-paste):

```
I have a local project that needs two YOLOv3-tiny files placed into its `models/` folder: `yolov3-tiny.cfg` (text) and `yolov3-tiny.weights` (binary). The repository intentionally excludes the weights because they're large. Please do one of the following (your choice), and explain the steps you take:

Option A — Produce a PowerShell script:
- Create a `download_models.ps1` script that:
  1. Creates the `models` directory if it does not exist.
  2. Downloads `yolov3-tiny.cfg` from https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg and saves it as `models\yolov3-tiny.cfg`.
  3. Downloads `yolov3-tiny.weights` from https://pjreddie.com/media/files/yolov3-tiny.weights and saves it as `models\yolov3-tiny.weights`.
  4. Optionally validates downloads by computing a SHA256 or file size and printing a summary.
- Output the full script content in your response and explain how to run it locally (PowerShell command).

Option B — Provide direct download instructions and verified official URLs:
- Output the exact commands for PowerShell and for a Unix `curl`/`wget` variant.
- Warn about the size of the weights file and suggest copying via USB if network access is unavailable.

Option C — If you have the ability to fetch files programmatically into my workspace (only do this if you can access the internet and are authorized), download both files and place them into the `models` folder, then report final file sizes.

Constraints:
- Do not attempt to generate or invent the binary contents of `yolov3-tiny.weights`. Only download from the official sources or provide the script/instructions.
- Ensure filenames match exactly: `yolov3-tiny.cfg`, `yolov3-tiny.weights`.
- If providing a script, use PowerShell that works on Windows 10+ and print helpful progress messages.

End of prompt.
```

If you would like, I can create the `download_models.ps1` script here in the repo (containing the commands above), so you can run it on your machine to fetch the files. Would you like me to add that script now? If yes, I will also add a short checksum step to the script to confirm file integrity.