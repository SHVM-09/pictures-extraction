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

# Timing decorator
def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print_info(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Core Functions for hashing and image comparison
def calculate_average_hash(image):
    """Generates an average hash for an image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL Image (RGB format)
    return str(imagehash.average_hash(pil_image))  # Generate average hash using imagehash and return as string

def calculate_perceptual_hash(image):
    """Generates a perceptual hash for an image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL Image (RGB format)
    return str(imagehash.phash(pil_image))  # Generate perceptual hash using imagehash and return as string

def calculate_difference_hash(image):
    """Generates a difference hash for an image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL Image (RGB format)
    return str(imagehash.dhash(pil_image))  # Generate difference hash using imagehash and return as string

def calculate_wavelet_hash(image):
    """Generates a wavelet hash for an image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL Image (RGB format)
    return str(imagehash.whash(pil_image))  # Generate wavelet hash using imagehash and return as string

def calculate_color_hash(image, binbits=3):
    """Generates a color hash for an image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL Image (RGB format)
    return str(imagehash.colorhash(pil_image, binbits=binbits))  # Generate color hash using imagehash and return as string

def calculate_crop_resistant_hash(image, min_segment_size=500, segmentation_image_size=1000):
    """Generates a crop-resistant hash for an image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL Image (RGB format)
    return str(imagehash.crop_resistant_hash(pil_image, min_segment_size=min_segment_size, segmentation_image_size=segmentation_image_size))  # Generate crop-resistant hash using imagehash

# Comparison Functions with time tracking
@time_function
def are_images_similar_phash(image1, image2, threshold=3):
    hash1 = calculate_perceptual_hash(image1)
    hash2 = calculate_perceptual_hash(image2)
    return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2) <= threshold

@time_function
def are_images_similar_dhash(image1, image2, threshold=3):
    hash1 = calculate_difference_hash(image1)
    hash2 = calculate_difference_hash(image2)
    return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2) <= threshold

@time_function
def are_images_similar_ahash(image1, image2, threshold=3):
    hash1 = calculate_average_hash(image1)
    hash2 = calculate_average_hash(image2)
    return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2) <= threshold

@time_function
def are_images_similar_whash(image1, image2, threshold=5):
    hash1 = calculate_wavelet_hash(image1)
    hash2 = calculate_wavelet_hash(image2)
    return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2) <= threshold

@time_function
def are_images_similar_colorhash(image1, image2, threshold=0):
    """Compares two images using colorhash and returns True if they are similar."""
    # Calculate the color hash for each image
    hash1 = calculate_color_hash(image1)
    hash2 = calculate_color_hash(image2)
    
    # Calculate the Hamming distance directly between the two hashes
    hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    # Return True if the Hamming distance is less than or equal to the threshold
    return hamming_distance <= threshold

@time_function
def are_images_similar_crophash(image1, image2, threshold=1000):
    """Compares two images using crop-resistant hash and returns True if they are similar."""
    # Calculate the crop-resistant hash for each image
    hash1 = calculate_crop_resistant_hash(image1)
    hash2 = calculate_crop_resistant_hash(image2)
    
    # Calculate the Hamming distance between the two hashes
    hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    # Return True if the Hamming distance is less than or equal to the threshold
    return hamming_distance <= threshold

@time_function
def are_images_similar_ssim(image1, image2, threshold=0.85):
    """Calculates SSIM (Structural Similarity Index) between two images."""
    # Convert both images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Normalize image values to range [0, 1]
    image1_gray /= 255.0
    image2_gray /= 255.0

    # Compute mean (mu) and standard deviation (sigma) for each image
    mu1 = cv2.GaussianBlur(image1_gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(image2_gray, (11, 11), 1.5)

    sigma1_sq = cv2.GaussianBlur(image1_gray ** 2, (11, 11), 1.5) - mu1 ** 2
    sigma2_sq = cv2.GaussianBlur(image2_gray ** 2, (11, 11), 1.5) - mu2 ** 2
    sigma12 = cv2.GaussianBlur(image1_gray * image2_gray, (11, 11), 1.5) - mu1 * mu2

    # Constants for stability in SSIM formula
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM formula
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    # Compute the SSIM value as the mean of the SSIM map
    ssim_value = np.mean(ssim_map)
    
    # Return True if SSIM >= threshold, meaning they are similar
    return ssim_value >= threshold

# Mean Squared Error (MSE)
@time_function
def are_images_similar_mse(image1, image2, threshold=1):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    mse_value = np.sum((image1_gray - image2_gray) ** 2) / float(image1_gray.shape[0] * image1_gray.shape[1])
    return mse_value <= threshold  # Lower MSE means more similarity

# Histogram Comparison (Correlation)
@time_function
def are_images_similar_hist(image1, image2, threshold=0.99):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])  # Calculate histogram for image1
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])  # Calculate histogram for image2
    hist1 = cv2.normalize(hist1, hist1).flatten()  # Normalize histogram
    hist2 = cv2.normalize(hist2, hist2).flatten()  # Normalize histogram
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  # Calculate correlation
    return correlation >= threshold  # Higher correlation means more similarity

# ORB (Oriented FAST and Rotated BRIEF) feature matching
@time_function
def are_images_similar_orb(image1, image2, threshold=1000):
    orb = cv2.ORB_create()  # Create ORB detector
    kp1, des1 = orb.detectAndCompute(image1, None)  # Detect keypoints and descriptors for image1
    kp2, des2 = orb.detectAndCompute(image2, None)  # Detect keypoints and descriptors for image2
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # Brute force matcher with Hamming distance
    matches = bf.match(des1, des2)  # Find matching keypoints
    matches = sorted(matches, key = lambda x: x.distance)  # Sort by distance
    match_ratio = len(matches) / (len(kp1) + len(kp2))  # Calculate match ratio
    return match_ratio >= threshold  # Higher ratio means more similarity

# Frame extraction with perceptual hash and additional image comparison methods

# Video Frame Extraction with Timing
@time_function
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
                # are_images_similar_dhash(frame, saved_frame)
                # are_images_similar_ahash(frame, saved_frame)
                # are_images_similar_whash(frame, saved_frame)
                # are_images_similar_colorhash(frame, saved_frame)
                # are_images_similar_crophash(frame, saved_frame)
                # are_images_similar_ssim(frame, saved_frame)
                # are_images_similar_mse(frame, saved_frame)
                # are_images_similar_hist(frame, saved_frame)
                are_images_similar_orb(frame, saved_frame)
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

# Audio Extraction with Transcription
@time_function
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
    output_folder = "output_frames_orb"  # Specify the output folder to save frames and transcript
    frame_interval = 30  # Process every 30th frame from the video

    print_heading("Video Processing Started")  # Display a heading message indicating that the process has started

    # Spinner control to show a loading spinner while processing
    spinner_event = threading.Event()

    spinner_thread = threading.Thread(target=spinner_task, args=("Extracting frames", spinner_event))
    spinner_thread.start()
    extracted_frames = extract_informative_frames(video_path, output_folder, frame_interval)
    spinner_event.set()
    spinner_thread.join()

    # Print the number of frames extracted
    num_frames_extracted = len(extracted_frames)
    print_info(f"Extracted {num_frames_extracted} frames: {extracted_frames}")

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
