import GPUtil
import torch


def get_gpu_with_most_memory():
    """
    Returns the CUDA device string for the GPU with the most available memory.

    Returns:
        str: Device string in format "cuda:x" where x is the GPU index

    Raises:
        RuntimeError: If no GPUs are available
    """
    gpus = GPUtil.getGPUs()

    if not gpus:
        if torch.backends.mps.is_available():
            print("No GPUs found, using Metal Performance Shaders (MPS) on macOS")
            return "mps"  # Use Metal Performance Shaders on macOS if available
        else:
            print("No GPUs found, falling back to CPU")
            return "cpu"  # Fallback to CPU if no GPUs are found

    # Find GPU with most free memory
    best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)

    # Optional: Print info about all GPUs
    print("Available GPUs:")
    for gpu in gpus:
        print(
            f"GPU {gpu.id}: {gpu.memoryFree:.0f} MB free out of {gpu.memoryTotal:.0f} MB total "
            f"({gpu.memoryUtil * 100:.1f}% used)"
        )

    print(f"\nSelected GPU {best_gpu.id} with {best_gpu.memoryFree:.0f} MB free memory")

    return f"cuda:{best_gpu.id}"


def get_gpu_with_most_memory_silent():
    """
    Returns the CUDA device string for the GPU with the most available memory (no output).

    Returns:
        str: Device string in format "cuda:x" where x is the GPU index

    Raises:
        RuntimeError: If no GPUs are available
    """
    gpus = GPUtil.getGPUs()

    if not gpus:
        if torch.backends.mps.is_available():
            return "mps"  # Use Metal Performance Shaders on macOS if available
        else:
            return "cpu"  # Fallback to CPU if no GPUs are found

    best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
    return f"cuda:{best_gpu.id}"


if __name__ == "__main__":
    try:
        # Verbose version
        best_device = get_gpu_with_most_memory()
        print(f"Best GPU device: {best_device}")

        # Or use the silent version
        # best_device = get_gpu_with_most_memory_silent()

    except RuntimeError as e:
        print(f"Error: {e}")
        print("Consider using CPU instead")
