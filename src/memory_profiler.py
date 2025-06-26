import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import gc
import psutil
import os
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
import warnings
import GPUtil

warnings.filterwarnings("ignore")

# Import your CNN model here
# from your_module import YourCNNModel


class ModelMemoryProfiler:
    def __init__(self, model_class, input_shape: Tuple[int, ...]):
        """
        Initialize the memory profiler.

        Args:
            model_class: Your CNN model class
            input_shape: Shape of input tensor (batch_size, channels, height, width)
        """
        self.model_class = model_class
        self.input_shape = input_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        self.num_timesteps = (
            1000  # Default number of timesteps, can be adjusted as needed
        )

        self.betas = self._cosine_beta_schedule(self.num_timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(
            self.device
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(self.device)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine noise schedule from "Improved Denoising Diffusion Probabilistic Models"

        Mathematical formulation:
        α̅_t = cos²((t/T + s)π/2 / (1 + s))
        β_t = 1 - α_t = 1 - α̅_t/α̅_{t-1}

        This schedule provides better sample quality than linear schedules.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: sample x_t from q(x_t | x_0)

        Mathematical formula:
        x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε

        Where ε ~ N(0, I) is Gaussian noise

        This is the reparameterization trick applied to:
        q(x_t | x_0) = N(x_t; √(α̅_t) x_0, (1-α̅_t) I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        if x_start.dim() == 2:
            # 2D data case (e.g., two moons, 2D Gaussian)
            reshape_dims = (-1, 1)
        elif x_start.dim() == 4:
            # 4D data case (e.g., images)
            reshape_dims = (-1, 1, 1, 1)
        else:
            # General case: reshape to match all dimensions except batch
            reshape_dims = (-1,) + (1,) * (x_start.dim() - 1)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(*reshape_dims)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            *reshape_dims
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory_info = {}

        # GPU memory if available
        if torch.cuda.is_available():
            memory_info["gpu_allocated_torch"] = (
                torch.cuda.memory_allocated() / 1024**2
            )  # MB
            memory_info["gpu_reserved_torch"] = (
                torch.cuda.memory_reserved() / 1024**2
            )  # MB
            memory_info["gpu_max_allocated_torch"] = (
                torch.cuda.max_memory_allocated() / 1024**2
            )  # MB

            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    memory_info["gpu_used_system"] = gpu.memoryUsed  # MB
                    memory_info["gpu_total_system"] = gpu.memoryTotal  # MB
                    memory_info["gpu_free_system"] = gpu.memoryFree  # MB
                    memory_info["gpu_util_percent"] = gpu.memoryUtil * 100  # %
            except Exception as e:
                print(f"Warning: GPUtil failed: {e}")

        # CPU memory
        process = psutil.Process(os.getpid())
        memory_info["cpu_memory"] = process.memory_info().rss / 1024**2  # MB

        return memory_info

    def clear_memory(self):
        """Clear GPU and CPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def test_configuration(
        self, hidden_dim: int, num_layers: int, batch_size: int
    ) -> Dict:
        """
        Test a single configuration and measure memory usage.

        Returns:
            Dictionary with results including memory usage or error info
        """
        config = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "batch_size": batch_size,
        }

        self.input_shape = (
            batch_size,
            3,
            256,
            256,
        )  # Update input shape based on batch size

        self.clear_memory()

        try:
            # Get baseline memory before model creation
            baseline_memory = self.get_memory_usage()

            # Create model with specified parameters
            model = self.model_class(
                input_channels=3,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                time_embed_dim=128,
                model_type="noise_predictor",
            )

            # Get memory after model creation (before moving to device)
            memory_after_creation = self.get_memory_usage()

            # Move model to device
            model = model.to(self.device)

            # Get memory after moving to device
            memory_after_device = self.get_memory_usage()

            # Create dummy input
            dummy_input = torch.randn(self.input_shape).to(self.device)
            dummy_time = torch.randint(
                0, self.num_timesteps, (self.input_shape[0],), device=self.device
            )
            dummy_noise = torch.randn_like(
                dummy_input, device=self.device
            )  # Assuming x_start is defined in your model

            # Get memory after input creation
            memory_after_input = self.get_memory_usage()

            # Get memory before forward pass
            memory_before = self.get_memory_usage()

            # Forward pass
            x_noisy = self.q_sample(dummy_input, dummy_time, noise=dummy_noise)
            output = model(x_noisy, dummy_time)
            loss = F.mse_loss(dummy_noise, output)
            loss.backward()

            # Get memory after forward pass
            memory_after = self.get_memory_usage()

            # Get memory after forward pass
            memory_after_forward = self.get_memory_usage()

            # Calculate memory differences
            result = config.copy()
            result["status"] = "success"

            # Model analysis
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            result["model_parameters"] = total_params
            result["trainable_parameters"] = trainable_params
            result["model_size_mb"] = total_params * 4 / 1024**2  # Assuming float32

            if torch.cuda.is_available():
                result["gpu_allocated_baseline"] = baseline_memory.get(
                    "gpu_allocated_torch", 0
                )
                result["gpu_allocated_after_creation"] = memory_after_creation.get(
                    "gpu_allocated_torch", 0
                )
                result["gpu_allocated_after_device"] = memory_after_device.get(
                    "gpu_allocated_torch", 0
                )
                result["gpu_allocated_after_input"] = memory_after_input.get(
                    "gpu_allocated_torch", 0
                )
                result["gpu_allocated_after_forward"] = memory_after_forward.get(
                    "gpu_allocated_torch", 0
                )

                # Memory deltas
                result["memory_delta_model_creation"] = memory_after_creation.get(
                    "gpu_allocated_torch", 0
                ) - baseline_memory.get("gpu_allocated_torch", 0)
                result["memory_delta_to_device"] = memory_after_device.get(
                    "gpu_allocated_torch", 0
                ) - memory_after_creation.get("gpu_allocated_torch", 0)
                result["memory_delta_input"] = memory_after_input.get(
                    "gpu_allocated_torch", 0
                ) - memory_after_device.get("gpu_allocated_torch", 0)
                result["memory_delta_forward"] = memory_after_forward.get(
                    "gpu_allocated_torch", 0
                ) - memory_after_input.get("gpu_allocated_torch", 0)

                result["gpu_memory_used_torch"] = memory_after[
                    "gpu_allocated_torch"
                ] - memory_before.get("gpu_allocated_torch", 0)
                result["gpu_peak_memory_torch"] = memory_after[
                    "gpu_max_allocated_torch"
                ]
                result["gpu_reserved_torch"] = memory_after["gpu_reserved_torch"]

                if "gpu_used_system" in memory_after:
                    result["gpu_memory_used_system"] = memory_after[
                        "gpu_used_system"
                    ] - memory_before.get("gpu_used_system", 0)
                    result["gpu_total_system"] = memory_after["gpu_total_system"]
                    result["gpu_util_percent"] = memory_after["gpu_util_percent"]

                    # System memory at each step
                    result["system_memory_baseline"] = baseline_memory.get(
                        "gpu_used_system", 0
                    )
                    result["system_memory_after_device"] = memory_after_device.get(
                        "gpu_used_system", 0
                    )
                    result["system_memory_after_forward"] = memory_after_forward.get(
                        "gpu_used_system", 0
                    )
                else:
                    result["gpu_memory_used_system"] = "N/A"
                    result["gpu_total_system"] = "N/A"
                    result["gpu_util_percent"] = "N/A"
            else:
                result["gpu_memory_used_torch"] = 0
                result["gpu_peak_memory_torch"] = 0
                result["gpu_reserved_torch"] = 0
                result["gpu_memory_used_system"] = 0
                result["gpu_total_system"] = "N/A"
                result["gpu_util_percent"] = 0

            result["cpu_memory"] = memory_after["cpu_memory"]
            result["output_shape"] = (
                str(output.shape) if hasattr(output, "shape") else "N/A"
            )

            # Debug: Print detailed info for first few configurations
            if len(self.results) < 3:  # Only for first 3 configurations
                print(f"\n  🔍 DEBUG INFO for config {config}:")
                print(
                    f"    Model params: {total_params:,} ({result['model_size_mb']:.1f} MB)"
                )
                if torch.cuda.is_available():
                    print(
                        f"    PyTorch GPU allocated: {result['gpu_allocated_baseline']:.1f} → {result['gpu_allocated_after_forward']:.1f} MB"
                    )
                    print(
                        f"    Memory deltas: creation={result['memory_delta_model_creation']:.1f}, "
                        f"to_device={result['memory_delta_to_device']:.1f}, "
                        f"input={result['memory_delta_input']:.1f}, "
                        f"forward={result['memory_delta_forward']:.1f}"
                    )
                    if result["gpu_memory_used_system"] != "N/A":
                        print(
                            f"    System GPU: {result['system_memory_baseline']:.1f} → {result['system_memory_after_forward']:.1f} MB"
                        )

            # Clean up
            del model, dummy_input, output, dummy_time, dummy_noise, loss, x_noisy

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                result = config.copy()
                result["status"] = "cuda_oom"
                result["error_message"] = str(e)
                result["model_parameters"] = "N/A"
                result["gpu_memory_used_torch"] = "N/A"
                result["gpu_peak_memory_torch"] = "N/A"
                result["gpu_reserved_torch"] = "N/A"
                result["gpu_memory_used_system"] = "N/A"
                result["gpu_total_system"] = "N/A"
                result["gpu_util_percent"] = "N/A"
                result["cpu_memory"] = "N/A"
                result["output_shape"] = "N/A"
            else:
                result = config.copy()
                result["status"] = "other_error"
                result["error_message"] = str(e)
                result["model_parameters"] = "N/A"
                result["gpu_memory_used_torch"] = "N/A"
                result["gpu_peak_memory_torch"] = "N/A"
                result["gpu_reserved_torch"] = "N/A"
                result["gpu_memory_used_system"] = "N/A"
                result["gpu_total_system"] = "N/A"
                result["gpu_util_percent"] = "N/A"
                result["cpu_memory"] = "N/A"
                result["output_shape"] = "N/A"

        except Exception as e:
            result = config.copy()
            result["status"] = "other_error"
            result["error_message"] = str(e)
            result["model_parameters"] = "N/A"
            result["gpu_memory_used_torch"] = "N/A"
            result["gpu_peak_memory_torch"] = "N/A"
            result["gpu_reserved_torch"] = "N/A"
            result["gpu_memory_used_system"] = "N/A"
            result["gpu_total_system"] = "N/A"
            result["gpu_util_percent"] = "N/A"
            result["cpu_memory"] = "N/A"
            result["output_shape"] = "N/A"

        finally:
            self.clear_memory()

        return result

    def _set_error_defaults(self, result: Dict):
        """Set default values for error cases."""
        error_fields = [
            "model_parameters",
            "trainable_parameters",
            "model_size_mb",
            "gpu_memory_used_torch",
            "gpu_peak_memory_torch",
            "gpu_reserved_torch",
            "gpu_memory_used_system",
            "gpu_total_system",
            "gpu_util_percent",
            "cpu_memory",
            "output_shape",
            "memory_delta_model_creation",
            "memory_delta_to_device",
            "memory_delta_input",
            "memory_delta_forward",
        ]
        for field in error_fields:
            result[field] = "N/A"

    def run_parameter_sweep(
        self, hidden_dims: List[int], num_layers_list: List[int], batch_sizes: List[int]
    ) -> pd.DataFrame:
        """
        Run parameter sweep across all combinations.

        Args:
            hidden_dims: List of hidden dimensions to test
            num_layers_list: List of number of layers to test
            time_embed_dims: List of time embedding dimensions to test

        Returns:
            DataFrame with all results
        """
        total_configs = len(hidden_dims) * len(num_layers_list) * len(batch_sizes)
        print(f"Testing {total_configs} configurations...")

        results = []
        for i, (hidden_dim, num_layers, batch_size) in enumerate(
            product(hidden_dims, num_layers_list, batch_sizes)
        ):
            print(
                f"Progress: {i + 1}/{total_configs} - Testing config: "
                f"hidden_dim={hidden_dim}, num_layers={num_layers}, batch_size={batch_size}"
            )

            result = self.test_configuration(hidden_dim, num_layers, batch_size)
            results.append(result)

            # Print status
            if result["status"] == "cuda_oom":
                print("  ❌ CUDA OOM Error")
            elif result["status"] == "success":
                memory_display = result.get("gpu_memory_used_torch", "N/A")
                if memory_display != "N/A":
                    print(f"  ✅ Success - PyTorch GPU Memory: {memory_display:.1f} MB")
                    if result.get("gpu_memory_used_system", "N/A") != "N/A":
                        print(
                            f"    System GPU Memory: {result['gpu_memory_used_system']:.1f} MB"
                        )
                else:
                    print("  ✅ Success")
            else:
                print(f"  ⚠️  Other Error: {result['status']}")
                print(f"    Error Message: {result.get('error_message', 'N/A')}")

        self.results = results
        return pd.DataFrame(results)

    def plot_results(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create comprehensive plots of the results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Memory Usage Parameter Sweep", fontsize=16)

        # Filter successful runs for memory plots
        success_df = df[df["status"] == "success"].copy()

        if len(success_df) > 0:
            # Convert memory to numeric for successful runs
            success_df["gpu_memory_torch_numeric"] = pd.to_numeric(
                success_df["gpu_memory_used_torch"], errors="coerce"
            )
            success_df["model_params_numeric"] = pd.to_numeric(
                success_df["model_parameters"], errors="coerce"
            )

            # Use system memory if available, otherwise torch memory
            if "gpu_memory_used_system" in success_df.columns:
                success_df["gpu_memory_system_numeric"] = pd.to_numeric(
                    success_df["gpu_memory_used_system"], errors="coerce"
                )
                # Use system memory for plots if available
                memory_column = "gpu_memory_system_numeric"
                memory_label = "System GPU Memory (MB)"
            else:
                memory_column = "gpu_memory_torch_numeric"
                memory_label = "PyTorch GPU Memory (MB)"

        # Plot 1: Success/Failure heatmap
        ax1 = axes[0, 0]
        pivot_status = df.pivot_table(
            values="status",
            index="hidden_dim",
            columns="num_layers",
            aggfunc=lambda x: (x == "success").sum(),
            fill_value=0,
        )
        sns.heatmap(pivot_status, annot=True, fmt="d", cmap="RdYlGn", ax=ax1)
        ax1.set_title("Success Count by Hidden Dim vs Num Layers")
        ax1.set_xlabel("Number of Layers")
        ax1.set_ylabel("Hidden Dimension")

        # Plot 2: GPU Memory Usage (successful runs only)
        ax2 = axes[0, 1]
        if len(success_df) > 0:
            pivot_memory = success_df.pivot_table(
                values=memory_column,
                index="hidden_dim",
                columns="num_layers",
                aggfunc="mean",
                fill_value=np.nan,
            )
            sns.heatmap(pivot_memory, annot=True, fmt=".1f", cmap="viridis", ax=ax2)
            ax2.set_title(f"Avg {memory_label}\nby Hidden Dim vs Num Layers")
        else:
            ax2.text(
                0.5,
                0.5,
                "No successful runs",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("GPU Memory Usage - No Data")

        # Plot 3: Model Parameters vs Memory
        ax3 = axes[1, 0]
        if len(success_df) > 0:
            scatter = ax3.scatter(
                success_df["model_params_numeric"],
                success_df[memory_column],
                c=success_df["num_layers"],
                cmap="plasma",
                alpha=0.7,
            )
            ax3.set_xlabel("Model Parameters")
            ax3.set_ylabel(memory_label)
            ax3.set_title("Memory vs Model Parameters")
            plt.colorbar(scatter, ax=ax3, label="Num Layers")
        else:
            ax3.text(
                0.5,
                0.5,
                "No successful runs",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title("Memory vs Parameters - No Data")

        # Plot 4: Error distribution
        ax4 = axes[1, 1]
        status_counts = df["status"].value_counts()
        colors = {"success": "green", "cuda_oom": "red", "other_error": "orange"}
        bar_colors = [colors.get(status, "gray") for status in status_counts.index]

        bars = ax4.bar(status_counts.index, status_counts.values, color=bar_colors)
        ax4.set_title("Configuration Status Distribution")
        ax4.set_ylabel("Count")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def save_results(self, df: pd.DataFrame, filepath: str):
        """Save results to CSV file."""
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")

    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "=" * 50)
        print("PARAMETER SWEEP SUMMARY")
        print("=" * 50)

        total_configs = len(df)
        successful = len(df[df["status"] == "success"])
        cuda_oom = len(df[df["status"] == "cuda_oom"])
        other_errors = len(df[df["status"] == "other_error"])

        print(f"Total configurations tested: {total_configs}")
        print(f"Successful: {successful} ({successful / total_configs * 100:.1f}%)")
        print(f"CUDA OOM errors: {cuda_oom} ({cuda_oom / total_configs * 100:.1f}%)")
        print(
            f"Other errors: {other_errors} ({other_errors / total_configs * 100:.1f}%)"
        )

        if successful > 0:
            success_df = df[df["status"] == "success"]

            # Determine which memory column to use for reporting
            if "gpu_memory_used_system" in success_df.columns:
                memory_col = "gpu_memory_used_system"
                memory_type = "System GPU Memory"
            else:
                memory_col = "gpu_memory_used_torch"
                memory_type = "PyTorch GPU Memory"

            memory_values = pd.to_numeric(
                success_df[memory_col], errors="coerce"
            ).dropna()

            if len(memory_values) > 0:
                print(f"\n{memory_type} usage for successful runs:")
                print(f"  Min: {memory_values.min():.1f} MB")
                print(f"  Max: {memory_values.max():.1f} MB")
                print(f"  Mean: {memory_values.mean():.1f} MB")

                # Show comparison if both are available
                if (
                    "gpu_memory_used_system" in success_df.columns
                    and "gpu_memory_used_torch" in success_df.columns
                ):
                    torch_memory = pd.to_numeric(
                        success_df["gpu_memory_used_torch"], errors="coerce"
                    ).dropna()
                    system_memory = pd.to_numeric(
                        success_df["gpu_memory_used_system"], errors="coerce"
                    ).dropna()
                    if len(torch_memory) > 0 and len(system_memory) > 0:
                        print("\nMemory reporting comparison:")
                        print(f"  PyTorch reports: {torch_memory.mean():.1f} MB (avg)")
                        print(f"  System reports: {system_memory.mean():.1f} MB (avg)")
                        print(
                            f"  Difference: {abs(system_memory.mean() - torch_memory.mean()):.1f} MB"
                        )

            print("\nModel parameters for successful runs:")
            params = pd.to_numeric(success_df["model_parameters"], errors="coerce")
            print(f"  Min: {params.min():,} parameters")
            print(f"  Max: {params.max():,} parameters")
            print(f"  Mean: {params.mean():,.0f} parameters")

    # Plotting functions
    def plot_3d_results(self, df: pd.DataFrame):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Success points
        success = df[df["status"] == "success"]
        print(success.head())
        sc = ax.scatter(
            success["hidden_dim"],
            success["num_layers"],
            success["batch_size"],
            c=success["gpu_memory_used_system"],
            cmap="viridis",
            s=50,
            label="Success",
        )

        # OOM points
        oom = df[df["status"] == "cuda_oom"]
        if not oom.empty:
            ax.scatter(
                oom["hidden_dim"],
                oom["num_layers"],
                oom["batch_size"],
                c="red",
                marker="X",
                s=100,
                label="OOM",
            )

        ax.set_xlabel("Hidden Dim")
        ax.set_ylabel("Num Layers")
        ax.set_zlabel("Batch size")
        ax.set_title("Model Memory Usage (MB)")
        fig.colorbar(sc, ax=ax, label="Memory (MB)")
        plt.legend()
        plt.savefig("3d_memory_usage.png", dpi=300)
        plt.close()


# Example usage
if __name__ == "__main__":
    # Example parameter ranges - adjust these based on your needs
    hidden_dims = [64, 128, 256, 512]
    num_layers_list = [2, 4, 6, 8, 10, 12, 20]
    batch_sizes = [32, 64, 128, 256, 512]

    # Input shape: (batch_size, channels, height, width)
    input_shape = (256, 3, 256, 256)  # Adjust based on your model's expected input

    # Uncomment and replace with your actual model import
    from utils.models import CNN

    profiler = ModelMemoryProfiler(CNN, input_shape)

    # For demonstration, using a placeholder
    df = profiler.run_parameter_sweep(hidden_dims, num_layers_list, batch_sizes)
    profiler.plot_results(df, "memory_analysis.png")
    profiler.save_results(df, "memory_results.csv")
    profiler.print_summary(df)
    profiler.plot_3d_results(df)
