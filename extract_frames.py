import cv2
import pytesseract
import os
import imagehash
from PIL import Image
import numpy as np
import subprocess
import whisper
import warnings
import itertools
import threading
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Styling Functions
def print_success(message):
    print(f"\033[92m✔ {message}\033[0m")  # Green text with a check mark

def print_info(message):
    print(f"\033[94mℹ {message}\033[0m")  # Blue text with an info icon

def print_warning(message):
    print(f"\033[93m⚠ {message}\033[0m")  # Yellow text with a warning icon

def print_error(message):
    print(f"\033[91m✖ {message}\033[0m")  # Red text with a cross icon

def print_heading(message):
    print(f"\033[1;95m{message}\033[0m")  # Bold magenta text

# Spinner Functionality
def spinner_task(message, spinner_event):
    spinner = itertools.cycle(["|", "/", "-", "\\"])
    while not spinner_event.is_set():
        print(f"\r{message} {next(spinner)}", end="", flush=True)
        time.sleep(0.1)
    print("\r" + " " * (len(message) + 2), end="", flush=True)

# Core Functions
def calculate_perceptual_hash(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(pil_image))

def are_images_similar(image1, image2, threshold=5):
    hash1 = calculate_perceptual_hash(image1)
    hash2 = calculate_perceptual_hash(image2)
    return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2) <= threshold

def extract_informative_frames(video_path, output_folder, frame_interval=30):
    print_heading("Extracting Informative Frames")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []
    final_frames = []

    if not cap.isOpened():
        print_error(f"Error: Unable to open video file: {video_path}")
        return []

    print_info(f"Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)

            if text.strip():
                is_duplicate = any(are_images_similar(frame, saved_frame) for saved_frame in final_frames)

                if not is_duplicate:
                    frame_file = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_file, frame)
                    saved_frames.append(frame_file)
                    final_frames.append(frame)

        frame_count += 1

    cap.release()
    print_success(f"Frames saved to: {output_folder}")
    return saved_frames

def extract_audio_to_text(video_path, output_folder):
    print_heading("Extracting and Transcribing Audio")
    audio_file = os.path.join(output_folder, "audio.wav")
    subprocess.run([
        "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_file, "-y"
    ], check=True)

    print_success(f"Audio extracted to: {audio_file}")

    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    transcript = result["text"]

    transcript_file = os.path.join(output_folder, "transcript.txt")
    with open(transcript_file, "w") as f:
        f.write(transcript)

    print_success(f"Transcript saved to: {transcript_file}")
    return transcript

# Main Script
if __name__ == "__main__":
    video_path = "videos/sample.mp4"
    output_folder = "output_frames"
    frame_interval = 30

    print_heading("Video Processing Started")

    # Spinner control
    spinner_event = threading.Event()

    # Step 1: Extract frames
    spinner_thread = threading.Thread(target=spinner_task, args=("Extracting frames", spinner_event))
    spinner_thread.start()
    extracted_frames = extract_informative_frames(video_path, output_folder, frame_interval)
    spinner_event.set()
    spinner_thread.join()
    print_info(f"Extracted frames: {extracted_frames}")

    # Step 2: Extract audio and generate transcript
    spinner_event.clear()
    spinner_thread = threading.Thread(target=spinner_task, args=("Processing audio and generating transcript", spinner_event))
    spinner_thread.start()
    transcript = extract_audio_to_text(video_path, output_folder)
    spinner_event.set()
    spinner_thread.join()
    print_info(f"Transcript: {transcript}")

    print_heading("Process Complete")
    print_success(f"Frames saved to: {output_folder}")
    print_success(f"Transcript saved to: {output_folder}/transcript.txt")
