import cv2
import os
import subprocess
import numpy as np
import whisper
import threading
import warnings
import time  # Importing time module for measuring execution time
from multiprocessing import Process, Queue

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
def split_video_into_chunks(video_path, chunk_size_mb=25):
    print_heading("Splitting Video into Chunks")

    video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if video_size_mb <= chunk_size_mb:
        print_info(f"Video size is already under {chunk_size_mb}MB. No need to split.")
        return [video_path]

    print_info(f"Splitting video: {video_path} into chunks of approximately {chunk_size_mb}MB each.")

    # Get total duration of the video
    total_duration = float(subprocess.check_output(
        ["ffprobe", "-i", video_path, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"],
        stderr=subprocess.PIPE
    ).strip())
    
    # Initial guess for chunk duration
    avg_bitrate = (video_size_mb * 8) / total_duration  # Mbps
    chunk_duration = (chunk_size_mb * 8) / avg_bitrate  # In seconds

    chunk_paths = []
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.dirname(video_path)
    start_time = 0
    chunk_index = 0

    while start_time < total_duration:
        chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{chunk_index:03d}.mp4")

        # Use ffmpeg to extract the chunk
        subprocess.run(
            [
                "ffmpeg", "-i", video_path, "-ss", str(start_time), "-t", str(chunk_duration),
                "-c", "copy", "-reset_timestamps", "1", chunk_path, "-y"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Validate the chunk size
        if os.path.exists(chunk_path):
            actual_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            if actual_size_mb > chunk_size_mb * 1.1:  # If chunk is too large, reduce duration
                chunk_duration *= 0.9
                os.remove(chunk_path)  # Delete the oversized chunk and retry
                continue
            elif actual_size_mb < chunk_size_mb * 0.9:  # If chunk is too small, increase duration
                chunk_duration *= 1.1

            chunk_paths.append(chunk_path)
            start_time += chunk_duration
            chunk_index += 1
        else:
            print_warning(f"Failed to create chunk: {chunk_path}")
            break

    print_success(f"Created {len(chunk_paths)} video chunks.")
    return chunk_paths

# Mean Squared Error (MSE)
def are_images_similar_mse(image1, image2, threshold=9):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    mse_value = np.sum((image1_gray - image2_gray) ** 2) / float(image1_gray.shape[0] * image1_gray.shape[1])
    return mse_value <= threshold  # Lower MSE means more similarity

# MSE-based frame extraction
@measure_time
def extract_informative_frames(video_path, output_folder, frame_interval=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)  # Ensure no error if directory exists

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []
    previous_frame = None

    if not cap.isOpened():
        print_error(f"Error: Unable to open video file: {video_path}")
        return []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            if previous_frame is None or not are_images_similar_mse(frame, previous_frame, threshold=9):
                frame_file = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_file, frame)
                saved_frames.append(frame_file)
                previous_frame = frame

        frame_count += 1

    cap.release()
    print_success(f"Extracted {len(saved_frames)} frames and saved them to {output_folder}")
    return saved_frames

# Validate audio chunks
def is_valid_audio_chunk(chunk):
    """
    Validates whether an audio chunk is non-empty and has a duration greater than 0 seconds.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-i", chunk, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        duration = float(result.stdout.strip())
        return duration > 0  # Valid if duration is greater than 0
    except Exception as e:
        print_warning(f"Invalid audio chunk {chunk}: {e}")
        return False


def transcribe_chunk(chunk, result_queue):
    """Worker function to transcribe an audio chunk."""
    try:
        model = whisper.load_model("base")  # Load a separate model for this process
        result = model.transcribe(chunk)
        result_queue.put((chunk, result["text"]))  # Send result back to main process
    except Exception as e:
        print_error(f"Error transcribing chunk {chunk}: {e}")
        result_queue.put((chunk, ""))  # Send empty result on error

@measure_time
def extract_audio_to_text(video_path, output_folder, audio_chunk_size_mb=25):
    print_heading("Extracting and Transcribing Audio")

    audio_file = os.path.join(output_folder, "audio.wav")
    print_info(f"Extracting audio from {video_path}")

    # Extract the audio from the video
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", "-q:a", "0", "-map", "a", audio_file, "-y"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to extract audio: {e}")
        return ""

    # Check audio size and split if necessary
    audio_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
    audio_chunks = []

    if audio_size_mb > audio_chunk_size_mb:
        print_info(f"Audio size exceeds {audio_chunk_size_mb}MB. Splitting audio into chunks.")
        chunk_output_pattern = os.path.join(output_folder, "audio_chunk_%03d.wav")
        subprocess.run(
            [
                "ffmpeg", "-i", audio_file, "-f", "segment", "-segment_time", "300",
                "-ar", "16000", "-ac", "1", chunk_output_pattern, "-y",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        audio_chunks = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.startswith("audio_chunk_")]
    else:
        audio_chunks = [audio_file]

    # Ensure non-empty and valid chunks
    valid_chunks = [chunk for chunk in audio_chunks if is_valid_audio_chunk(chunk)]
    if not valid_chunks:
        print_error("No valid audio chunks found for transcription.")
        return ""

    print_info(f"Transcribing {len(valid_chunks)} valid audio chunks.")

    # Use multiprocessing for parallel transcription
    processes = []
    result_queue = Queue()

    for chunk in sorted(valid_chunks):
        process = Process(target=transcribe_chunk, args=(chunk, result_queue))
        processes.append(process)
        process.start()

    # Collect results
    transcripts = []
    for _ in range(len(valid_chunks)):
        transcripts.append(result_queue.get())

    # Ensure all processes finish
    for process in processes:
        process.join()

    # Sort transcripts by chunk order and combine
    transcripts.sort(key=lambda x: x[0])  # Sort by chunk filename to maintain order
    final_transcript = " ".join([text for _, text in transcripts])

    transcript_file = os.path.join(output_folder, "transcript.txt")
    with open(transcript_file, "w") as f:
        f.write(final_transcript)

    print_success(f"Transcript saved to: {transcript_file}")

    # Cleanup audio chunks and temporary files
    for chunk in audio_chunks:
        if os.path.exists(chunk):
            os.remove(chunk)
    if os.path.exists(audio_file):
        os.remove(audio_file)

    return final_transcript

# Process chunks in parallel
def process_chunk(chunk_path, output_folder, frame_interval, results):
    frames = extract_informative_frames(chunk_path, output_folder, frame_interval)
    results["frames"].extend(frames)

# Main function to handle video processing
def process_large_video(video_path, output_folder, frame_interval=30):
    print_heading("Processing Video")
    total_start_time = time.time()  # Start timer for the entire script

    # Split video into chunks
    chunks = split_video_into_chunks(video_path)

    # Extract frames
    results = {"frames": []}
    print_heading("Extracting Informative Frames")
    frame_start_time = time.time()
    threads = []
    for chunk_path in chunks:
        thread = threading.Thread(target=process_chunk, args=(chunk_path, output_folder, frame_interval, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    frame_elapsed_time = time.time() - frame_start_time
    print_info(f"Total time taken for frame extraction: {frame_elapsed_time:.2f} seconds.")

    # Audio to text transcription
    audio_start_time = time.time()
    transcript = extract_audio_to_text(video_path, output_folder)
    audio_elapsed_time = time.time() - audio_start_time
    print_info(f"Total time taken for audio transcription: {audio_elapsed_time:.2f} seconds.")

    # Cleanup video chunks
    for chunk_path in chunks:
        if chunk_path != video_path:
            os.remove(chunk_path)

    total_elapsed_time = time.time() - total_start_time
    print_success(f"Frames and transcript saved in: {output_folder}")
    print_success(f"Total time taken for the entire script: {total_elapsed_time:.2f} seconds.")
    return transcript

if __name__ == "__main__":
    video_path = "videos/sample.mp4"
    output_folder = "output_frames_mse"
    frame_interval = 30

    process_large_video(video_path, output_folder, frame_interval)
