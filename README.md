Here’s a detailed, **line-by-line explanation** of the code:

---

### **Imports**
```python
import cv2
import pytesseract
import os
import imagehash
from PIL import Image
import numpy as np
```
1. `cv2`: OpenCV library for computer vision tasks, used here for video frame extraction and image manipulation.
2. `pytesseract`: Python wrapper for the Tesseract OCR tool, used to detect text in frames.
3. `os`: Python's library for file system operations (e.g., creating directories, joining paths).
4. `imagehash`: Library for calculating perceptual hashes, used to identify visually similar images.
5. `PIL.Image`: PIL (Python Imaging Library) module for handling image processing tasks.
6. `numpy`: Python library for working with arrays, used here to represent images as arrays.

---

### **Print Tesseract Version**
```python
print(pytesseract.get_tesseract_version())
```
This prints the version of Tesseract OCR installed on your system, ensuring it's properly set up.

---

### **Perceptual Hash Calculation**
```python
def calculate_perceptual_hash(image):
    """
    Calculate a perceptual hash for the given image.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL format
    return str(imagehash.phash(pil_image))  # Compute the perceptual hash
```
1. **Input**: `image` is a frame from the video represented as a NumPy array.
2. **`cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`**: Converts the image from OpenCV's BGR format to RGB.
3. **`Image.fromarray(...)`**: Converts the NumPy array to a PIL image.
4. **`imagehash.phash(...)`**: Calculates a perceptual hash using the `phash` method, which is robust to small visual changes like noise.
5. **Output**: Returns a string representation of the hash.

---

### **Image Similarity Check**
```python
def are_images_similar(image1, image2, threshold=5):
    """
    Check if two images are similar based on perceptual hash distance.
    """
    hash1 = calculate_perceptual_hash(image1)  # Compute hash for the first image
    hash2 = calculate_perceptual_hash(image2)  # Compute hash for the second image
    return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2) <= threshold  # Compare hash distances
```
1. **Input**: Two images (`image1`, `image2`) and a similarity threshold.
2. **`calculate_perceptual_hash(...)`**: Calculates perceptual hashes for both images.
3. **`imagehash.hex_to_hash(...)`**: Converts hash strings back to hash objects for comparison.
4. **Hamming Distance**: Measures the difference between the two hashes.
   - If the distance is less than or equal to `threshold`, the images are considered similar.
5. **Output**: Returns `True` if the images are similar, `False` otherwise.

---

### **Extract Informative Frames**
```python
def extract_informative_frames(video_path, output_folder, frame_interval=30):
    """
    Extract frames with information from a video and save them as distinct images.
    """
```
1. **Input Parameters**:
   - `video_path`: Path to the input video file.
   - `output_folder`: Directory to save the extracted frames.
   - `frame_interval`: Process every `frame_interval`-th frame (to reduce redundancy).

---

#### **Create Output Directory**
```python
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
```
1. Checks if the `output_folder` exists.
2. If not, it creates the folder using `os.makedirs`.

---

#### **Open the Video**
```python
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []
    final_frames = []
```
1. **`cv2.VideoCapture(video_path)`**: Opens the video file for processing.
2. **`frame_count`**: Tracks the current frame number.
3. **`saved_frames`**: Stores paths of saved frames.
4. **`final_frames`**: Stores actual frame data for comparison.

---

#### **Handle Errors**
```python
    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        return []
```
1. Checks if the video file was successfully opened.
2. If not, prints an error message and exits the function.

---

#### **Process Video Frames**
```python
    while True:
        ret, frame = cap.read()
        if not ret:
            break
```
1. **`cap.read()`**: Reads the next frame from the video.
   - `ret`: Boolean indicating success.
   - `frame`: The actual image frame.
2. **Exit Condition**: Stops when there are no more frames (`ret` is `False`).

---

#### **Process Every `frame_interval`-th Frame**
```python
        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
```
1. Skips frames to process only every `frame_interval`-th frame.
2. **`cv2.cvtColor(...)`**: Converts the frame to grayscale.
3. **`pytesseract.image_to_string(...)`**: Uses OCR to extract text from the grayscale image.

---

#### **Filter Frames with Text**
```python
            if text.strip():
                is_duplicate = any(are_images_similar(frame, saved_frame) for saved_frame in final_frames)
                if not is_duplicate:
                    frame_file = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_file, frame)
                    saved_frames.append(frame_file)
                    final_frames.append(frame)
```
1. Checks if the extracted text (`text.strip()`) is non-empty.
2. **Duplicate Check**:
   - Compares the current frame with all previously finalized frames using `are_images_similar`.
3. **Save Frame**:
   - If not a duplicate, saves the frame to the `output_folder`.
   - Updates `saved_frames` and `final_frames`.

---

#### **Increment Frame Counter**
```python
        frame_count += 1
```
Increments the frame counter to keep track of the current frame number.

---

#### **Release Resources**
```python
    cap.release()
    print(f"Frames saved to: {output_folder}")
    return saved_frames
```
1. **`cap.release()`**: Closes the video file.
2. Prints the location of saved frames.
3. Returns the list of saved frame paths.

---

### **Main Program**
```python
if __name__ == "__main__":
    video_path = "videos/sample.mp4"
    output_folder = "output_frames"
    frame_interval = 30

    extracted_frames = extract_informative_frames(video_path, output_folder, frame_interval)
    print(f"Extracted frames: {extracted_frames}")
```
1. Sets up the input video path and output folder.
2. Calls `extract_informative_frames` with the provided arguments.
3. Prints the list of saved frame paths after processing.

---

### **Summary**
- **Core Functionality**:
  - Extracts frames containing text from a video.
  - Ensures only unique, distinct frames are saved.
- **Key Features**:
  - Skips redundant frames (`frame_interval`).
  - Detects text in frames (`pytesseract`).
  - Filters duplicates using perceptual hashing (`imagehash`).

Let me know if you'd like further clarification or improvements!