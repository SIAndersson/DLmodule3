import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from dotenv import load_dotenv
from pytorch_lightning.callbacks import Callback
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, kstest, wasserstein_distance
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.neighbors import NearestNeighbors

load_dotenv()

DATASET_CACHE = os.getenv("HF_DATASETS_CACHE", None)


# Taken from Reddit to avoid scipy sqrtm https://www.reddit.com/r/MachineLearning/comments/12hv2u6/d_a_better_way_to_compute_the_fr%C3%A9chet_inception/
def frechet_distance(
    mu_x: torch.Tensor, sigma_x: torch.Tensor, mu_y: torch.Tensor, sigma_y: torch.Tensor
):
    a = (mu_x - mu_y).square().sum(dim=-1)
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum(dim=-1)

    return a + b - 2 * c


# Taken from https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    """Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# The cooler Daniel (the faster callback)
class FastEvaluationCallback(Callback):
    """
    Fast evaluation callback for generative models.
    Computes lightweight metrics during training without slowing down significantly.
    """

    def __init__(
        self,
        model_type="vector_field",  # vector_field or diffusion
        dataset_type="2d",  # 2d or image
        eval_every_n_epochs=5,
        num_samples=1000,
        batch_size=100,
        feature_extractor="mobilenet",  # 'mobilenet', 'resnet18', or None
        cache_dir="./model_cache",
        compute_coverage_precision=True,
        compute_mmd=True,
        compute_wasserstein=True,
        compute_js_divergence=True,
        compute_energy_distance=True,
        compute_density_consistency=True,
        compute_mode_collapse=True,
        compute_spectral_metrics=True,
        compute_diversity_metrics=True,
        k_nearest=5,  # for coverage/precision
        mmd_kernel="rbf",  # 'rbf' or 'polynomial'
        device="auto",
    ):
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.eval_every_n_epochs = eval_every_n_epochs
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.feature_extractor_name = feature_extractor
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.compute_coverage_precision = compute_coverage_precision
        self.compute_mmd = compute_mmd
        self.compute_wasserstein = compute_wasserstein
        self.compute_js_divergence = compute_js_divergence
        self.compute_energy_distance = compute_energy_distance
        self.compute_density_consistency = compute_density_consistency
        self.compute_mode_collapse = compute_mode_collapse
        self.compute_spectral_metrics = compute_spectral_metrics
        self.compute_diversity_metrics = compute_diversity_metrics
        self.k_nearest = k_nearest
        self.mmd_kernel = mmd_kernel
        self.device = device

        self.feature_extractor = None
        self.real_features_cache = None
        self.real_samples_cache = None

        # Store metrics history
        self.metrics_history = {
            "epoch": [],
            "wasserstein_distance": [],
            "mmd": [],
            "coverage": [],
            "precision": [],
            "js_divergence": [],
            "energy_distance": [],
            "density_ks_stat": [],
            "log_density_ratio": [],
            "mode_collapse_score": [],
            "duplicate_ratio": [],
            "spectral_divergence": [],
            "condition_number_ratio": [],
            "mean_pairwise_distance": [],
            "min_pairwise_distance": [],
            "std_pairwise_distance": [],
            "distance_entropy": [],
        }

    def setup(self, trainer, pl_module, stage=None):
        """Setup feature extractor and cache real data features"""
        if self.device == "auto":
            self.device = pl_module.device

        # Setup feature extractor if needed
        if (
            self.feature_extractor_name and pl_module.input_dim > 2
        ):  # Assume images if dim > 2
            self._setup_feature_extractor()

    def _setup_feature_extractor(self):
        """Initialize lightweight feature extractor"""
        if self.feature_extractor_name == "mobilenet":
            # Download to specified cache directory
            os.environ["TORCH_HOME"] = str(self.cache_dir)
            model = models.mobilenet_v2(pretrained=True)
            # Remove classifier to get features
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 1280

        elif self.feature_extractor_name == "resnet18":
            os.environ["TORCH_HOME"] = str(self.cache_dir)
            model = models.resnet18(pretrained=True)
            # Remove final FC layer
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 512

        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)

        # Image preprocessing
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _extract_features(self, samples):
        """Extract features from samples"""
        if self.feature_extractor is None:
            return samples.view(samples.shape[0], -1)  # Flatten

        features = []
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i : i + self.batch_size]

            # Ensure proper format for image models
            if batch.dim() == 4 and batch.shape[1] == 3:  # RGB images
                batch = self.preprocess(batch)
            elif batch.dim() == 4 and batch.shape[1] == 1:  # Grayscale
                batch = batch.repeat(1, 3, 1, 1)  # Convert to RGB
                batch = self.preprocess(batch)

            with torch.no_grad():
                feat = self.feature_extractor(batch)
                feat = feat.view(feat.size(0), -1)  # Flatten
                features.append(feat.cpu())

        return torch.cat(features, dim=0)

    def _cache_real_data(self, trainer):
        """Cache real training data features"""
        if self.real_features_cache is not None:
            return

        print("Caching real data features...")
        real_samples = []

        # Sample from training data
        train_loader = trainer.train_dataloader
        samples_collected = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            real_samples.append(x.cpu())
            samples_collected += x.shape[0]

            if samples_collected >= self.num_samples:
                break

        real_samples = torch.cat(real_samples, dim=0)[: self.num_samples]
        self.real_samples_cache = real_samples

        # Extract features
        real_samples = real_samples.to(self.device)
        self.real_features_cache = self._extract_features(real_samples).cpu()
        print(f"Cached {len(self.real_features_cache)} real samples")

    def _generate_samples(self, pl_module):
        """Generate samples from the model"""
        pl_module.eval()
        generated_samples = []

        with torch.no_grad():
            for _ in range(0, self.num_samples, self.batch_size):
                current_batch_size = min(
                    self.batch_size,
                    self.num_samples - len(generated_samples) * self.batch_size,
                )

                if self.dataset_type == "2d":
                    if self.model_type == "vector_field":
                        samples = pl_module.fast_sample(current_batch_size, self.device)
                    elif self.model_type == "diffusion":
                        samples = pl_module.ddim_sample(
                            (current_batch_size, 2), self.device
                        )
                    else:
                        raise ValueError(f"Unknown model type: {self.model_type}")
                else:  # image
                    if self.model_type == "vector_field":
                        samples = pl_module.fast_sample(current_batch_size, self.device)
                    elif self.model_type == "diffusion":
                        samples = pl_module.ddim_sample(
                            (current_batch_size, 3, 264, 264), self.device
                        )
                    else:
                        raise ValueError(f"Unknown model type: {self.model_type}")
                    generated_samples.append(samples.cpu())

        return torch.cat(generated_samples, dim=0)

    def _compute_wasserstein_distance(self, real_features, fake_features):
        """Compute 1-Wasserstein distance"""
        if real_features.shape[1] > 1:
            # For high-dimensional data, compute per-dimension and average
            distances = []
            for dim in range(
                min(real_features.shape[1], 10)
            ):  # Limit to first 10 dims for speed
                wd = wasserstein_distance(
                    real_features[:, dim].numpy(), fake_features[:, dim].numpy()
                )
                distances.append(wd)
            return np.mean(distances)
        else:
            return wasserstein_distance(
                real_features.flatten().numpy(), fake_features.flatten().numpy()
            )

    def _compute_mmd(self, real_features, fake_features, kernel="rbf"):
        """Compute Maximum Mean Discrepancy"""
        X = real_features.numpy()
        Y = fake_features.numpy()

        if kernel == "rbf":
            # Use multiple bandwidths for better results
            bandwidths = [0.1, 1.0, 10.0]
            mmd_vals = []

            for bw in bandwidths:
                K_XX = rbf_kernel(X, X, gamma=1 / (2 * bw**2))
                K_YY = rbf_kernel(Y, Y, gamma=1 / (2 * bw**2))
                K_XY = rbf_kernel(X, Y, gamma=1 / (2 * bw**2))

                mmd_val = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
                mmd_vals.append(mmd_val)

            return np.mean(mmd_vals)

        elif kernel == "polynomial":
            K_XX = polynomial_kernel(X, X, degree=2)
            K_YY = polynomial_kernel(Y, Y, degree=2)
            K_XY = polynomial_kernel(X, Y, degree=2)

            return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

    def _compute_coverage_precision(self, real_features, fake_features, k=5):
        """Compute Coverage and Precision metrics using k-NN"""
        real_np = real_features.numpy()
        fake_np = fake_features.numpy()

        # Fit k-NN on real data
        real_nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        real_nn.fit(real_np)

        # Fit k-NN on fake data
        fake_nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        fake_nn.fit(fake_np)

        # Coverage: fraction of real samples whose k-NN sphere contains at least one fake sample
        real_distances, _ = real_nn.kneighbors(real_np)
        real_radii = real_distances[:, k]  # k-th nearest neighbor distance

        fake_to_real_distances, _ = fake_nn.kneighbors(real_np)
        fake_to_real_min_dist = fake_to_real_distances[:, 1]  # closest fake sample

        coverage = (fake_to_real_min_dist < real_radii).mean()

        # Precision: fraction of fake samples whose k-NN sphere contains at least one real sample
        fake_distances, _ = fake_nn.kneighbors(fake_np)
        fake_radii = fake_distances[:, k]

        real_to_fake_distances, _ = real_nn.kneighbors(fake_np)
        real_to_fake_min_dist = real_to_fake_distances[:, 1]

        precision = (real_to_fake_min_dist < fake_radii).mean()

        return coverage, precision

    def _compute_js_divergence(self, real_features, fake_features, bins=50):
        """Compute Jensen-Shannon divergence between feature distributions"""
        js_divs = []

        # Compute for each feature dimension (limit to first 10 for speed)
        n_dims = min(real_features.shape[1], 10)

        for dim in range(n_dims):
            real_vals = real_features[:, dim].numpy()
            fake_vals = fake_features[:, dim].numpy()

            # Create histograms with same bins
            min_val = min(real_vals.min(), fake_vals.min())
            max_val = max(real_vals.max(), fake_vals.max())

            if max_val == min_val:
                js_divs.append(0.0)
                continue

            bins_edges = np.linspace(min_val, max_val, bins + 1)

            real_hist, _ = np.histogram(real_vals, bins=bins_edges, density=True)
            fake_hist, _ = np.histogram(fake_vals, bins=bins_edges, density=True)

            # Add small epsilon to avoid log(0)
            eps = 1e-10
            real_hist = real_hist + eps
            fake_hist = fake_hist + eps

            # Normalize
            real_hist = real_hist / real_hist.sum()
            fake_hist = fake_hist / fake_hist.sum()

            js_div = jensenshannon(real_hist, fake_hist, base=2)
            js_divs.append(js_div)

        return np.mean(js_divs)

    def _compute_energy_distance(self, real_features, fake_features):
        """Compute Energy Distance between two samples"""
        X = real_features.numpy()
        Y = fake_features.numpy()

        # Subsample for efficiency if datasets are large
        if len(X) > 500:
            idx_x = np.random.choice(len(X), 500, replace=False)
            X = X[idx_x]
        if len(Y) > 500:
            idx_y = np.random.choice(len(Y), 500, replace=False)
            Y = Y[idx_y]

        # Compute pairwise distances
        def pairwise_distances(A, B):
            return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))

        # E[|X-Y|]
        dXY = pairwise_distances(X, Y).mean()

        # E[|X-X'|]
        dXX = pairwise_distances(X, X).mean()

        # E[|Y-Y'|]
        dYY = pairwise_distances(Y, Y).mean()

        energy_distance = 2 * dXY - dXX - dYY
        return energy_distance

    def _compute_density_consistency(self, real_features, fake_features, k=5):
        """Measure how well fake samples match real data density using k-NN density estimation"""
        real_np = real_features.numpy()
        fake_np = fake_features.numpy()

        # Fit k-NN on real data for density estimation
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(real_np)

        # Get k-NN distances for real samples (exclude self)
        real_distances, _ = nn.kneighbors(real_np)
        real_densities = 1.0 / (real_distances[:, k] + 1e-10)  # k-th neighbor distance

        # Get k-NN distances for fake samples
        fake_distances, _ = nn.kneighbors(fake_np)
        fake_densities = 1.0 / (fake_distances[:, k] + 1e-10)

        # Compare density distributions using KS test
        ks_stat, _ = kstest(fake_densities, real_densities)

        # Also compute density ratio statistics
        density_ratio = np.log(fake_densities.mean() / real_densities.mean() + 1e-10)

        return {"density_ks_stat": ks_stat, "log_density_ratio": density_ratio}

    def _compute_mode_collapse_metrics(self, fake_features, k=10):
        """Detect mode collapse using intra-cluster distance and nearest neighbor analysis"""
        fake_np = fake_features.numpy()

        if len(fake_np) < k:
            return {"mode_collapse_score": 0.0, "duplicate_ratio": 0.0}

        # Fit k-NN on fake samples
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(fake_np)

        distances, indices = nn.kneighbors(fake_np)

        # Mode collapse score: average distance to k-th nearest neighbor
        # Lower values indicate potential mode collapse
        knn_distances = distances[:, k]  # k-th neighbor distance
        mode_collapse_score = knn_distances.mean()

        # Duplicate detection: samples with very small nearest neighbor distance
        duplicate_threshold = np.percentile(
            distances[:, 1], 5
        )  # 5th percentile of 1-NN distances
        duplicate_ratio = (distances[:, 1] < duplicate_threshold).mean()

        return {
            "mode_collapse_score": mode_collapse_score,
            "duplicate_ratio": duplicate_ratio,
        }

    def _compute_spectral_metrics(self, real_features, fake_features):
        """Compute spectral properties of the feature matrices"""
        real_np = real_features.numpy()
        fake_np = fake_features.numpy()

        # Center the data
        real_centered = real_np - real_np.mean(axis=0)
        fake_centered = fake_np - fake_np.mean(axis=0)

        # Compute covariance matrices
        real_cov = np.cov(real_centered.T)
        fake_cov = np.cov(fake_centered.T)

        # Compute eigenvalues
        real_eigenvals = np.linalg.eigvals(real_cov)
        fake_eigenvals = np.linalg.eigvals(fake_cov)

        # Sort eigenvalues in descending order
        real_eigenvals = np.sort(real_eigenvals)[::-1]
        fake_eigenvals = np.sort(fake_eigenvals)[::-1]

        # Take only positive eigenvalues and limit to top 10 for speed
        real_eigenvals = real_eigenvals[real_eigenvals > 1e-10][:10]
        fake_eigenvals = fake_eigenvals[fake_eigenvals > 1e-10][:10]

        # Spectral divergence (compare eigenvalue distributions)
        min_len = min(len(real_eigenvals), len(fake_eigenvals))
        if min_len > 0:
            spectral_divergence = np.mean(
                np.abs(
                    np.log(real_eigenvals[:min_len] + 1e-10)
                    - np.log(fake_eigenvals[:min_len] + 1e-10)
                )
            )
        else:
            spectral_divergence = float("inf")

        # Condition number ratio
        real_condition = (
            real_eigenvals[0] / (real_eigenvals[-1] + 1e-10)
            if len(real_eigenvals) > 0
            else 1.0
        )
        fake_condition = (
            fake_eigenvals[0] / (fake_eigenvals[-1] + 1e-10)
            if len(fake_eigenvals) > 0
            else 1.0
        )
        condition_ratio = np.log(fake_condition / (real_condition + 1e-10))

        return {
            "spectral_divergence": spectral_divergence,
            "condition_number_ratio": condition_ratio,
        }

    def _compute_diversity_metrics(self, fake_features):
        """Compute diversity metrics for generated samples"""
        fake_np = fake_features.numpy()

        # Pairwise distances between generated samples
        def pairwise_l2_distances(X):
            diff = X[:, None, :] - X[None, :, :]
            return np.sqrt((diff**2).sum(axis=2))

        # Subsample for efficiency
        if len(fake_np) > 500:
            idx = np.random.choice(len(fake_np), 500, replace=False)
            fake_subset = fake_np[idx]
        else:
            fake_subset = fake_np

        distances = pairwise_l2_distances(fake_subset)

        # Remove diagonal (self-distances)
        mask = ~np.eye(distances.shape[0], dtype=bool)
        distances_flat = distances[mask]

        # Diversity metrics
        mean_pairwise_distance = distances_flat.mean()
        min_pairwise_distance = distances_flat.min()
        std_pairwise_distance = distances_flat.std()

        # Effective sample size (entropy of distance distribution)
        hist, _ = np.histogram(distances_flat, bins=50, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        distance_entropy = entropy(hist, base=2)

        return {
            "mean_pairwise_distance": mean_pairwise_distance,
            "min_pairwise_distance": min_pairwise_distance,
            "std_pairwise_distance": std_pairwise_distance,
            "distance_entropy": distance_entropy,
        }

    def on_train_epoch_end(self, trainer, pl_module):
        """Run evaluation at the end of train epoch"""
        if trainer.current_epoch % self.eval_every_n_epochs != 0:
            return

        # Cache real data on first evaluation
        self._cache_real_data(trainer)

        # Generate samples
        print(f"Generating {self.num_samples} samples for evaluation...")
        generated_samples = self._generate_samples(pl_module)

        # Extract features
        generated_samples = generated_samples.to(self.device)
        fake_features = self._extract_features(generated_samples).cpu()

        metrics = {}

        # Compute Wasserstein distance
        if self.compute_wasserstein:
            wd = self._compute_wasserstein_distance(
                self.real_features_cache, fake_features
            )
            metrics["wasserstein_distance"] = wd

        # Compute MMD
        if self.compute_mmd:
            mmd = self._compute_mmd(
                self.real_features_cache, fake_features, self.mmd_kernel
            )
            metrics["mmd"] = mmd

        # Compute Coverage and Precision
        if self.compute_coverage_precision:
            coverage, precision = self._compute_coverage_precision(
                self.real_features_cache, fake_features, self.k_nearest
            )
            metrics["coverage"] = coverage
            metrics["precision"] = precision

        # Compute Jensen-Shannon Divergence
        if self.compute_js_divergence:
            js_div = self._compute_js_divergence(
                self.real_features_cache, fake_features
            )
            metrics["js_divergence"] = js_div

        # Compute Energy Distance
        if self.compute_energy_distance:
            energy_dist = self._compute_energy_distance(
                self.real_features_cache, fake_features
            )
            metrics["energy_distance"] = energy_dist

        # Compute Density Consistency
        if self.compute_density_consistency:
            density_metrics = self._compute_density_consistency(
                self.real_features_cache, fake_features
            )
            metrics.update(density_metrics)

        # Compute Mode Collapse Metrics
        if self.compute_mode_collapse:
            mode_metrics = self._compute_mode_collapse_metrics(fake_features)
            metrics.update(mode_metrics)

        # Compute Spectral Metrics
        if self.compute_spectral_metrics:
            spectral_metrics = self._compute_spectral_metrics(
                self.real_features_cache, fake_features
            )
            metrics.update(spectral_metrics)

        # Compute Diversity Metrics
        if self.compute_diversity_metrics:
            diversity_metrics = self._compute_diversity_metrics(fake_features)
            metrics.update(diversity_metrics)

        # Log metrics
        for name, value in metrics.items():
            pl_module.log(f"eval/{name}", value)

        # Store in history
        self.metrics_history["epoch"].append(trainer.current_epoch)
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append(value)

    def get_metrics_history(self):
        """Return the complete metrics history"""
        return self.metrics_history


class MetricTracker(Callback):
    def __init__(self):
        self.train_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses.append(loss.item())


def create_evaluation_callback(
    cfg, model_type="diffusion", evaluation_level="standard"
):
    """Create appropriate evaluation callback"""

    if cfg.main.dataset.lower() in ["two_moons", "2d_gaussians"]:
        data_type = "2d"
    else:
        data_type = "image"

    if evaluation_level == "minimal":
        # Fastest evaluation - core metrics only
        return FastEvaluationCallback(
            model_type=model_type,
            dataset_type=data_type,
            eval_every_n_epochs=5,
            num_samples=500,
            feature_extractor="mobilenet" if data_type == "image" else None,
            cache_dir="./weights",
            compute_coverage_precision=True,
            compute_mmd=True,
            compute_wasserstein=False,  # Skip for speed
            compute_js_divergence=False,
            compute_energy_distance=False,
            compute_density_consistency=False,
            compute_mode_collapse=True,  # Important for generative models
            compute_spectral_metrics=False,
            compute_diversity_metrics=False,
        )

    elif evaluation_level == "comprehensive":
        # Full evaluation suite
        return FastEvaluationCallback(
            model_type=model_type,
            dataset_type=data_type,
            eval_every_n_epochs=10,
            num_samples=2000,
            feature_extractor="mobilenet" if data_type == "image" else None,
            cache_dir="./weights",
            compute_coverage_precision=True,
            compute_mmd=True,
            compute_wasserstein=True,
            compute_js_divergence=True,
            compute_energy_distance=True,
            compute_density_consistency=True,
            compute_mode_collapse=True,
            compute_spectral_metrics=True,
            compute_diversity_metrics=True,
        )

    else:  # 'standard'
        if data_type == "image":
            return FastEvaluationCallback(
                model_type=model_type,
                dataset_type=data_type,
                eval_every_n_epochs=10,  # Less frequent for images
                num_samples=1000,  # Moderate sample size
                feature_extractor="mobilenet",
                cache_dir="./weights",
                compute_coverage_precision=True,
                compute_mmd=True,
                compute_wasserstein=True,
                compute_js_divergence=True,
                compute_energy_distance=False,  # Skip for images (expensive)
                compute_density_consistency=True,
                compute_mode_collapse=True,
                compute_spectral_metrics=True,
                compute_diversity_metrics=True,
            )
        else:  # 2D or low-dimensional data
            return FastEvaluationCallback(
                model_type=model_type,
                dataset_type=data_type,
                eval_every_n_epochs=5,
                num_samples=2000,  # More samples for better statistics
                feature_extractor=None,  # No feature extraction needed
                compute_coverage_precision=True,
                compute_mmd=True,
                compute_wasserstein=True,
                compute_js_divergence=True,
                compute_energy_distance=True,
                compute_density_consistency=True,
                compute_mode_collapse=True,
                compute_spectral_metrics=True,
                compute_diversity_metrics=True,
            )
