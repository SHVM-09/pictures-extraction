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
pip install opencv-python pytesseract pillow imagehash
```

This command installs:
- `opencv-python`: For video processing and frame extraction.
- `pytesseract`: For Optical Character Recognition (OCR) on frames.
- `pillow`: For image processing (used with `imagehash`).
- `imagehash`: For comparing image similarity.

---

### **3. Install and Configure Tesseract OCR**
1. **Install Tesseract**:
   On macOS:
   ```bash
   brew install tesseract
   ```

2. **Verify Tesseract Installation**:
   Check the version to confirm the installation:
   ```bash
   tesseract --version
   ```

---

### **4. Prepare the Video File**
1. Create a folder named `videos` in the same directory as the script.
2. Place your video file (e.g., `sample.mp4`) in the `videos` folder.

---

### **5. Run the Script**
Execute the script using Python:
```bash
python3 extract_frames.py
```

---

### **Commands to Automate Installation and Run**
If you'd like to automate the setup, use the following commands:

#### **One-Liner for Installation**
```bash
python3 -m venv myenv && source myenv/bin/activate && pip install opencv-python pytesseract pillow imagehash && brew install tesseract
```

#### **Run the Script**
```bash
python3 extract_frames.py
```

---

### **Expected Output**
1. Extracted frames with text will be saved in the `output_frames` folder.
2. The console will display:
   - The frames processed.
   - The saved frames and their paths.

---