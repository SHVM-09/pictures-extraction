### Frame Comparison Methods Performance Table

| **Method**        | **Frames Processed** | **Processing Time (secs)** | **Performance Analysis**                                                                                          |
|--------------------|-----------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------|
| **aHash**          | 31                   | 7.98                       | ✅ Fast and efficient. Good for comparing images with similar overall structures but may miss subtle changes.      |
| **colorHash**      | 16                   | 16.79                      | ⚠ Slow and limited in handling minor frame changes. Less reliable for real-world use cases.                       |
| **dHash**          | 38                   | 9.26                       | ✅ Fast and detects minor differences well. Ideal for slight modifications but slower than MSE.                   |
| **Histogram**      | 19                   | 5.43                       | ⚠ Fast but less accurate. Tends to neglect small changes and unsuitable for detailed comparison tasks.            |
| **MSE**            | 52                   | 4.25                       | ✅ Fastest method. Accurate for pixel-level comparisons but misses perceptual differences.                         |
| **ORB**            | 117                  | 89.75                      | ❌ Extremely slow. Inefficient for large datasets and prone to detecting unnecessary duplicates.                   |
| **pHash**          | 45                   | 11.76                      | ✅ Robust to perceptual changes. Slightly slower but ideal for detecting content-based changes.                    |
| **SSIM**           | 15                   | 10.17                      | ⚠ Slow and misses subtle changes. Focuses on structural similarity but unsuitable for high-volume processing.      |
| **wHash**          | 43                   | 44.42                      | ❌ Too slow and computationally intensive for large-scale use.                                                     |
| **cropHash**       | N/A                  | Crashed                   | ❌ Computationally impractical and extremely time-consuming.                                                       |

---

### **Key Insights**
1. **Best Performing Methods**:
   - **MSE**: Ideal for high-speed processing with minimal computational cost.
   - **pHash**: Best for detecting perceptual changes in content, albeit slightly slower.

2. **Methods to Avoid**:
   - **ORB, wHash, cropHash**: Inefficient due to high computational time or impracticality.
   - **SSIM, Histogram**: Lack robustness for real-world applications where small changes matter.

---

### **Recommendation**
- Use **MSE** for speed-critical tasks with minimal perceptual differences.
- Use **pHash** for scenarios requiring detection of subtle changes, such as color or texture variations.

---

### **1. Threading**
- **Definition**: Threading allows multiple threads (lightweight units of a process) to run concurrently within the same process.
- **Context in the Code**:
  ```python
  frame_process = threading.Thread(target=extract_informative_frames,
                                   args=(video_chunks, output_folder, frame_interval, video_creation_time))
  audio_process = threading.Thread(target=extract_audio_to_text,
                                   args=(video_path, output_folder, video_creation_time))
  frame_process.start()
  audio_process.start()
  frame_process.join()
  audio_process.join()
  ```
  - **Usage**:
    - Two threads are created: one for extracting informative frames and the other for audio-to-text transcription.
    - These threads share the same memory space (the main process).
  - **Key Features**:
    - Threads are lightweight and created quickly.
    - Useful when tasks are I/O-bound (e.g., reading/writing files, waiting for external data like audio or frames).
    - Threads share memory and data structures, simplifying communication between them.
  - **Limitation**:
    - In Python, the **Global Interpreter Lock (GIL)** prevents multiple threads from executing Python bytecode simultaneously. This means only one thread runs at a time when performing CPU-bound tasks.
    - In this case, threading is suitable because the tasks involve **I/O operations** (e.g., file reading/writing), not heavy computation.

---

### **2. Multiprocessing**
- **Definition**: Multiprocessing spawns multiple processes, each with its own Python interpreter and memory space, to run concurrently.
- **Context in the Code**:
  ```python
  with Pool(processes=cpu_count(), maxtasksperchild=1) as pool:
      pool.map(extract_frames_from_chunk, pool_args)
  ```
  - **Usage**:
    - A `Pool` of worker processes is created to parallelize the task of extracting frames across multiple video chunks.
    - Each process works independently and has its own memory space.
  - **Key Features**:
    - Multiprocessing bypasses the GIL, allowing true parallelism, which is ideal for CPU-bound tasks (e.g., heavy computations, image processing).
    - Processes do not share memory by default; data must be passed explicitly between them, typically using inter-process communication (IPC) mechanisms.
  - **Limitation**:
    - Processes are heavier and slower to create than threads.
    - Inter-process communication introduces additional overhead, making multiprocessing less efficient for I/O-bound tasks.

---

### **Key Differences in the Code Context**

| **Aspect**             | **Threading**                                                    | **Multiprocessing**                                            |
|------------------------|------------------------------------------------------------------|----------------------------------------------------------------|
| **Use Case**           | I/O-bound tasks (e.g., extracting audio, writing files).        | CPU-bound tasks (e.g., extracting frames, comparing images).   |
| **Execution**          | Runs within the same process; limited by the GIL.              | Creates separate processes; no GIL limitation.                |
| **Memory Sharing**     | Threads share memory and data structures.                      | Processes have separate memory spaces; data sharing requires IPC. |
| **Overhead**           | Lightweight; lower overhead.                                   | Higher overhead due to process creation and memory isolation.  |
| **Parallelism**        | Concurrency only for I/O-bound tasks (not true parallelism).    | True parallelism for CPU-bound tasks.                         |

---

### **Why the Code Uses Both?**

1. **Threading**:
   - Used for **concurrent I/O-bound tasks** (`extract_informative_frames` and `extract_audio_to_text`):
     - Both involve reading/writing files and interacting with external processes (`ffmpeg`), making threading ideal.
   - Threads allow these tasks to overlap, optimizing execution time.

2. **Multiprocessing**:
   - Used for **CPU-intensive frame processing** (`extract_frames_from_chunk`):
     - Extracting frames and comparing images (`are_images_similar_mse`) are computationally expensive.
     - Multiprocessing enables **parallel execution** across multiple CPU cores, significantly speeding up these tasks.

---

### **Summary**
- **Threading**: Best for I/O-bound tasks where the GIL is not a limitation.
- **Multiprocessing**: Best for CPU-bound tasks where true parallelism is needed.
- The code combines both to optimize resource usage, leveraging threading for I/O tasks and multiprocessing for CPU-intensive tasks.