import cv2  # Import OpenCV for video processing and frame extraction
# import pytesseract  # Optional, not used in the current version
import os  # For file and folder handling
import imagehash  # For perceptual hashing of images
from PIL import Image  # For image manipulation using PIL
import numpy as np  # For numerical operations (not directly used in this script)
import subprocess  # For running external commands (e.g., ffmpeg to extract audio)
import whisper  # For speech-to-text transcription using the Whisper model
import warnings  # For suppressing warnings
import itertools  # For cyclic iteration (used in spinner)
import threading  # For multi-threading to show spinner during processing
import time  # For time-related tasks (e.g., delays)

# Suppress warnings
warnings.filterwarnings("ignore")  # Ignore any warnings that might be generated during processing

# Styling Functions to print messages in different colors and formats for better visibility
def print_success(message):
    """Prints a success message in green color with a check mark."""
    print(f"\033[92m✔ {message}\033[0m")  # Green text with a check mark

def print_info(message):
    """Prints an informational message in blue color with an info icon."""
    print(f"\033[94mℹ {message}\033[0m")  # Blue text with an info icon

def print_warning(message):
    """Prints a warning message in yellow color with a warning icon."""
    print(f"\033[93m⚠ {message}\033[0m")  # Yellow text with a warning icon

def print_error(message):
    """Prints an error message in red color with a cross icon."""
    print(f"\033[91m✖ {message}\033[0m")  # Red text with a cross icon

def print_heading(message):
    """Prints a bold heading message in magenta color."""
    print(f"\033[1;95m{message}\033[0m")  # Bold magenta text for headings

# Spinner Functionality to show a loading spinner during long operations
def spinner_task(message, spinner_event):
    """Displays a spinner animation during long-running tasks."""
    spinner = itertools.cycle(["|", "/", "-", "\\"])  # Spinner characters
    while not spinner_event.is_set():  # Continue showing spinner until the event is set
        print(f"\r{message} {next(spinner)}", end="", flush=True)  # Print spinner with message
        time.sleep(0.1)  # Wait for 0.1 seconds before changing spinner character
    print("\r" + " " * (len(message) + 2), end="", flush=True)  # Clear the spinner

# Core Functions for video frame extraction and audio transcription

def calculate_perceptual_hash(image):
    """Generates a perceptual hash for an image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL Image (RGB format)
    return str(imagehash.phash(pil_image))  # Generate perceptual hash using imagehash and return as string

def are_images_similar_phash(image1, image2, threshold=3):
    hash1 = calculate_perceptual_hash(image1)
    hash2 = calculate_perceptual_hash(image2)
    return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2) <= threshold

# Mean Squared Error (MSE)
def are_images_similar_mse(image1, image2, threshold=1):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    mse_value = np.sum((image1_gray - image2_gray) ** 2) / float(image1_gray.shape[0] * image1_gray.shape[1])
    return mse_value <= threshold  # Lower MSE means more similarity

# Frame extraction with perceptual hash and additional image comparison methods
def extract_informative_frames(video_path, output_folder, frame_interval=30):
    """
    Extracts frames from a video at specified intervals and saves them to an output folder.
    Frames that are similar (based on perceptual hash) are skipped.
    """
    print_heading("Extracting Informative Frames")  # Display a heading message for this process
    if not os.path.exists(output_folder):  # Check if the output folder exists, if not, create it
        os.makedirs(output_folder)  # Create the output folder

    cap = cv2.VideoCapture(video_path)  # Open the video file using OpenCV
    frame_count = 0  # Initialize frame count
    saved_frames = []  # List to store the file paths of saved frames
    final_frames = []  # List to store the actual frames to compare similarity

    if not cap.isOpened():  # Check if the video file could be opened
        print_error(f"Error: Unable to open video file: {video_path}")  # Print error if video file cannot be opened
        return []  # Return an empty list in case of error

    print_info(f"Processing video: {video_path}")  # Inform the user that video processing has started

    while True:
        ret, frame = cap.read()  # Read the next frame from the video
        if not ret:  # If the frame could not be read (end of video)
            break  # Exit the loop

        if frame_count % frame_interval == 0:
            is_duplicate = any(
                # are_images_similar_phash(frame, saved_frame)
                are_images_similar_mse(frame, saved_frame)
                for saved_frame in final_frames)

            if not is_duplicate:  # If the frame is not a duplicate, save it
                frame_file = os.path.join(output_folder, f"frame_{frame_count}.jpg")  # Generate file name for the frame
                cv2.imwrite(frame_file, frame)  # Save the frame as a .jpg image file
                saved_frames.append(frame_file)  # Add the saved frame file path to the list
                final_frames.append(frame)  # Add the actual frame to the list for future comparison

        frame_count += 1  # Increment frame count

    cap.release()  # Release the video capture object when done
    print_success(f"Frames saved to: {output_folder}")  # Inform the user that frames have been saved
    return saved_frames  # Return the list of saved frames' file paths

def extract_audio_to_text(video_path, output_folder):
    """
    Extracts the audio from a video and transcribes it into text using the Whisper model.
    Deletes the audio file after transcription.
    """
    print_heading("Extracting and Transcribing Audio")  # Display a heading message for audio extraction and transcription
    audio_file = os.path.join(output_folder, "audio.wav")  # Specify the path for the audio file to be saved
    
    # Extract the audio from the video using ffmpeg
    subprocess.run([
        "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_file, "-y"
    ], check=True)  # Run ffmpeg to extract audio as a .wav file

    print_success(f"Audio extracted to: {audio_file}")  # Inform the user that audio extraction is complete

    # Use Whisper (pre-trained speech-to-text model) to transcribe the audio file
    model = whisper.load_model("base")  # Load the Whisper model (base model is lightweight)
    result = model.transcribe(audio_file)  # Transcribe the audio to text
    transcript = result["text"]  # Extract the text from the transcription result

    # Save the transcript to a text file
    transcript_file = os.path.join(output_folder, "transcript.txt")
    with open(transcript_file, "w") as f:
        f.write(transcript)  # Write the transcript text to the file

    print_success(f"Transcript saved to: {transcript_file}")  # Inform the user that the transcript has been saved

    # Delete the audio file after transcription to save space
    os.remove(audio_file)  # Remove the audio file from the disk
    print_success(f"Deleted audio file: {audio_file}")  # Inform the user that the audio file has been deleted

    return transcript  # Return the transcript text

# Main Script that runs the video processing and transcription tasks

if __name__ == "__main__":
    video_path = "videos/sample.mp4"  # Specify the path to the input video file
    output_folder = "output_frames_mse"  # Specify the output folder to save frames and transcript
    frame_interval = 30  # Process every 30th frame from the video

    print_heading("Video Processing Started")  # Display a heading message indicating that the process has started

    # Spinner control to show a loading spinner while processing
    spinner_event = threading.Event()

    # Step 1: Extract frames from the video
    spinner_thread = threading.Thread(target=spinner_task, args=("Extracting frames", spinner_event))  # Start spinner in a separate thread
    spinner_thread.start()  # Start spinner thread
    extracted_frames = extract_informative_frames(video_path, output_folder, frame_interval)  # Call the frame extraction function
    spinner_event.set()  # Stop the spinner once frame extraction is done
    spinner_thread.join()  # Wait for spinner thread to finish
    print_info(f"Extracted frames: {extracted_frames}")  # Print information about the extracted frames

    # Step 2: Extract audio and generate transcript
    spinner_event.clear()  # Clear spinner event to start new spinner for audio processing
    spinner_thread = threading.Thread(target=spinner_task, args=("Processing audio and generating transcript", spinner_event))  # Start spinner for audio processing
    spinner_thread.start()  # Start spinner thread
    transcript = extract_audio_to_text(video_path, output_folder)  # Call the audio extraction and transcription function
    spinner_event.set()  # Stop the spinner once transcription is done
    spinner_thread.join()  # Wait for spinner thread to finish
    print_info(f"Transcript: {transcript}")  # Print the transcript generated from the audio

    # Final messages indicating the process is complete
    print_heading("Process Complete")  # Display a heading indicating that the process is complete
    print_success(f"Frames saved to: {output_folder}")  # Inform the user that frames have been saved
    print_success(f"Transcript saved to: {output_folder}/transcript.txt")  # Inform the user that the transcript has been saved
