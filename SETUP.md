### **Steps to Run the Code**

Follow the steps below to set up the environment, install the necessary packages, and execute the script.

---

### **1. Set Up Python Environment**
1. **Ensure Python is Installed**:
   Check if Python is installed:
   ```bash
   python3 --version
   ```
   If Python is not installed, install it using Homebrew:
   ```bash
   brew install python
   ```

2. **Create a Virtual Environment**:
   Set up a virtual environment to isolate your dependencies:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```

---

### **2. Install Required Packages**
Run the following command to install all the necessary dependencies:

```bash
pip install opencv-python numpy openai-whisper
```

```bash
brew install ffmpeg
```

`ffmpeg` is required for video and audio processing.

If warnings/error in whisper use command  `pip install --upgrade openai-whisper`

---

### **3. Prepare the Video File**
1. Create a folder named `videos` in the same directory as the script.
2. Place your video file (e.g., `sample.mp4`) in the `videos` folder.

---

### **4. Run the Script**
Execute the script using Python:
```bash
python3 extract_frames_timestamp.py
```

---

### **Expected Output**

1. **Extracted Frames**:
   - Extracted frames are saved in the `outputs` folder.
   - Each frame is named with its timestamp (e.g., `frames-<timestamp>.jpg`).

2. **Transcripts**:
   - Audio transcriptions are saved as `.txt` files in the `outputs` folder.
   - Each file is named with its timestamp (e.g., `transcript-<timestamp>.txt`).

3. **Terminal**:
   - The script will display detailed logs, including:
     - Frames processed.
     - Audio transcription details.
     - Total time taken.