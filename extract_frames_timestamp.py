import cv2
import os
import subprocess
import numpy as np
import whisper
import threading
import warnings
import time  # Importing time module for measuring execution time
from multiprocessing import Pool, cpu_count

# Suppress warnings
warnings.filterwarnings("ignore")  # Ignore any warnings that might be generated during processing

# Styling Functions to print messages in different colors and formats for better visibility
def print_success(message):
    """Prints a success message in green color with a check mark."""
    print(f"\033[92m✔ {message}\033[0m")  # Green text with a check mark and newline

def print_info(message):
    """Prints an informational message in blue color with an info icon."""
    print(f"\033[94mℹ {message}\033[0m")  # Blue text with an info icon and newline

def print_warning(message):
    """Prints a warning message in yellow color with a warning icon."""
    print(f"\033[93m⚠ {message}\033[0m")  # Yellow text with a warning icon and newline

def print_error(message):
    """Prints an error message in red color with a cross icon."""
    print(f"\033[91m✖ {message}\033[0m")  # Red text with a cross icon and newline

def print_heading(message):
    """Prints a bold heading message in magenta color."""
    print(f"\033[1;95m{message}\033[0m")  # Bold magenta text for headings with a newline

# Add timer decorator to measure execution time of functions
def measure_time(func):
    """Decorator to measure and print execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print_info(f"{func.__name__} took {elapsed_time:.2f} seconds.")
        return result
    return wrapper

# Split video into chunks of ~25MB
@measure_time
def split_video_into_chunks(video_path, output_folder):
    """Splits the video into 1-minute chunks using ffmpeg."""
    print_heading("Splitting Video into 1-Minute Chunks")
    chunk_output_pattern = os.path.join(output_folder, "video_chunk_%03d.mp4")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-c", "copy", "-map", "0", "-f", "segment",
         "-segment_time", "60", "-y", chunk_output_pattern],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    chunks = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder)
                     if f.startswith("video_chunk_") and f.endswith(".mp4")])
    print_success(f"Created {len(chunks)} video chunks.")
    return chunks

def extract_frames_from_chunk(args):
    """Processes a single video chunk to extract frames."""
    chunk, output_folder, frame_interval, global_offset_time, index = args
    print_info(f"Processing chunk: {chunk}")
    cap = cv2.VideoCapture(chunk)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Handle invalid FPS
    frame_count, previous_frame = 0, None
    saved_frames = []

    # Offset based on chunk index
    chunk_offset_time = global_offset_time + (index * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            elapsed_seconds = frame_count / fps
            frame_timestamp = int(chunk_offset_time + elapsed_seconds)
            if previous_frame is None or not are_images_similar_mse(frame, previous_frame):
                frame_file = os.path.join(output_folder, f"frames-{frame_timestamp}.jpg")
                cv2.imwrite(frame_file, frame)
                saved_frames.append(frame_file)
                previous_frame = frame
        frame_count += 1

    cap.release()
    os.remove(chunk)  # Cleanup chunk
    print_success(f"Extracted {len(saved_frames)} frames from chunk: {chunk}")

def are_images_similar_mse(image1, image2, threshold=9):
    """Compares two images using Mean Squared Error."""
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    mse_value = np.sum((image1_gray - image2_gray) ** 2) / float(image1_gray.shape[0] * image1_gray.shape[1])
    return mse_value <= threshold

@measure_time
def extract_informative_frames(video_chunks, output_folder, frame_interval, video_creation_time):
    """Extracts frames across video chunks using multiprocessing."""
    print_heading("Extracting Informative Frames")
    os.makedirs(output_folder, exist_ok=True)
    pool_args = [(chunk, output_folder, frame_interval, video_creation_time, idx)
                 for idx, chunk in enumerate(video_chunks)]

    with Pool(processes=cpu_count()) as pool:
        pool.map(extract_frames_from_chunk, pool_args)

@measure_time
def extract_audio_to_text(video_path, output_folder, video_creation_time):
    """Extracts audio, splits into chunks, and transcribes."""
    print_heading("Extracting and Transcribing Audio")
    audio_file = os.path.join(output_folder, "audio.wav")
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", "-q:a", "0", "-map", "a", audio_file, "-y"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    chunk_pattern = os.path.join(output_folder, "audio_chunk_%03d.wav")
    subprocess.run(
        ["ffmpeg", "-i", audio_file, "-f", "segment", "-segment_time", "60", chunk_pattern, "-y"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    audio_chunks = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder)
                           if f.startswith("audio_chunk_")])
    args_list = [(chunk, int(video_creation_time + (idx * 60)), output_folder)
                 for idx, chunk in enumerate(audio_chunks)]

    with Pool(min(cpu_count(), 4)) as pool:
        pool.map(transcribe_chunk_with_timestamp, args_list)

    for chunk in audio_chunks:
        os.remove(chunk)
    os.remove(audio_file)
    print_success("Audio transcription completed and audio chunks deleted.")

def transcribe_chunk_with_timestamp(args):
    """Worker function to transcribe a single audio chunk."""
    chunk_path, chunk_timestamp, output_folder = args
    try:
        model = whisper.load_model("base")
        result = model.transcribe(chunk_path)
        transcript_file = os.path.join(output_folder, f"transcript-{chunk_timestamp}.txt")
        with open(transcript_file, "w") as f:
            f.write(result["text"])
        print_success(f"Transcript saved: {transcript_file}")
    except Exception as e:
        print_error(f"Error transcribing {chunk_path}: {e}")

def process_large_video(video_path, output_folder, frame_interval=30):
    print_heading("Processing Video")
    os.makedirs(output_folder, exist_ok=True)
    video_creation_time = os.path.getctime(video_path)

    # Step 1: Split video into chunks
    video_chunks = split_video_into_chunks(video_path, output_folder)

    # Step 2: Extract frames and transcribe audio concurrently
    start_time = time.time()
    frame_process = threading.Thread(target=extract_informative_frames,
                                     args=(video_chunks, output_folder, frame_interval, video_creation_time))
    audio_process = threading.Thread(target=extract_audio_to_text,
                                     args=(video_path, output_folder, video_creation_time))
    frame_process.start()
    audio_process.start()
    frame_process.join()
    audio_process.join()

    total_time = time.time() - start_time
    print_success(f"Total processing time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    script_start_time = time.time()  # Start timer for the entire script
    video_path = "videos/check.mov"
    output_folder = "outputs"
    frame_interval = 30
    process_large_video(video_path, output_folder, frame_interval)
    total_time = time.time() - script_start_time  # End timer
    print_success(f"Total script execution time: {total_time:.2f} seconds.")
