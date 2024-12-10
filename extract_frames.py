import cv2
import pytesseract
import os
import imagehash
from PIL import Image
import numpy as np

print(pytesseract.get_tesseract_version())


def calculate_perceptual_hash(image):
    """
    Calculate a perceptual hash for the given image.
    
    Args:
        image (numpy.ndarray): Image to hash.
    
    Returns:
        str: Perceptual hash of the image.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(pil_image))


def are_images_similar(image1, image2, threshold=5):
    """
    Check if two images are similar based on perceptual hash distance.
    
    Args:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.
        threshold (int): Maximum allowed Hamming distance between hashes to consider images as similar.
    
    Returns:
        bool: True if images are similar, False otherwise.
    """
    hash1 = calculate_perceptual_hash(image1)
    hash2 = calculate_perceptual_hash(image2)
    return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2) <= threshold


def extract_informative_frames(video_path, output_folder, frame_interval=30):
    """
    Extract frames with information from a video and save them as distinct images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Folder to save the extracted frames.
        frame_interval (int): Process every `frame_interval`-th frame.

    Returns:
        List[str]: List of file paths to the saved frames.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []
    final_frames = []  # To store frames for similarity comparison

    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        return []

    print(f"Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every `frame_interval`-th frame
        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)

            # Save frame only if it contains text
            if text.strip():
                # Check for similarity with previously finalized frames
                is_duplicate = any(are_images_similar(frame, saved_frame) for saved_frame in final_frames)

                if not is_duplicate:
                    # Save the frame
                    frame_file = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_file, frame)
                    saved_frames.append(frame_file)
                    final_frames.append(frame)  # Add frame to comparison list

        frame_count += 1

    cap.release()
    print(f"Frames saved to: {output_folder}")
    return saved_frames


# Example Usage
if __name__ == "__main__":
    video_path = "videos/sample.mp4"  # Path to the demo video
    output_folder = "output_frames"      # Folder to save extracted frames
    frame_interval = 30                  # Process every 30th frame

    extracted_frames = extract_informative_frames(video_path, output_folder, frame_interval)
    print(f"Extracted frames: {extracted_frames}")
