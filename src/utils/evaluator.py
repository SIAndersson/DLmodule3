import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from scipy.stats import wasserstein_distance, kstest, entropy
from pathlib import Path
import os


class StandaloneGenerativeModelEvaluator:
    """
    Standalone evaluator for generative models. Caches real data features once,
    then allows multiple evaluations against different generated samples.
    """

    def __init__(
        self,
        logger,
        dataset_type="image",  # 2d or image
        feature_extractor="inception",  # 'mobilenet', 'inception', or None
        cache_dir="./weights",
        compute_coverage_precision=True,
        compute_mmd=True,
        compute_wasserstein=True,
        compute_js_divergence=True,
        compute_energy_distance=True,
        compute_density_consistency=True,
        compute_mode_collapse=True,
        compute_diversity_metrics=True,
        compute_fid=True,  # Lightweight FID for image datasets
        k_nearest=5,  # for coverage/precision
        mmd_kernel="rbf",  # 'rbf' or 'polynomial'
        batch_size=128,  # for feature extraction
    ):
        self.logger = logger
        self.dataset_type = dataset_type
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
        self.compute_diversity_metrics = compute_diversity_metrics
        self.compute_fid = compute_fid
        self.k_nearest = k_nearest
        self.mmd_kernel = mmd_kernel

        self.feature_extractor = None
        self.real_features_cache = None
        self.device = None

    def setup_feature_extractor(self, device):
        """Initialize lightweight feature extractor"""
        if not self.feature_extractor_name or self.dataset_type == "2d":
            return

        if self.compute_fid or self.feature_extractor_name == "inception":
            os.environ["TORCH_HOME"] = str(self.cache_dir)

            # Load Inception-v3 pretrained on ImageNet (standard for FID)
            model = models.inception_v3(pretrained=True, transform_input=False)
            self.logger.info("Loaded Inception.")

            # Remove the final classification layers to get features from pool3
            # This gives us the 2048-dimensional feature vector used in FID
            model.fc = nn.Identity()
            model.AuxLogits.fc = nn.Identity()

            for param in model.parameters():
                param.requires_grad = False

            self.feature_extractor = model
            self.feature_dim = 2048

        elif self.feature_extractor_name == "mobilenet":
            os.environ["TORCH_HOME"] = str(self.cache_dir)
            model = models.mobilenet_v2(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 1280

        if self.feature_extractor:
            self.feature_extractor.eval()
            self.feature_extractor.to(device)

            if self.feature_extractor_name == "inception":
                input_size = 299
            else:
                input_size = 224

            self.preprocess = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.logger.info("Set up transforms.")

    def _extract_features(self, samples):
        """Extract features from samples"""
        if self.feature_extractor is None:
            return samples.view(samples.shape[0], -1)  # Flatten

        features = []
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i : i + self.batch_size]

            # Handle different input formats and ensure correct preprocessing
            if batch.dim() == 4:
                # Check if values are in [0,1] or [-1,1] range and convert to [0,1] if needed (Inception wants [-1, 1] so do not convert)
                self.logger.info(
                    f"BEFORE TRANSFORM: Batch min {batch.min()}. Batch max {batch.max()}."
                )
                if batch.min() < 0 and self.feature_extractor_name != "inception":
                    batch = (batch + 1) / 2  # Convert from [-1,1] to [0,1]

                # Ensure RGB format
                if batch.shape[1] == 1:  # Grayscale
                    batch = batch.repeat(1, 3, 1, 1)  # Convert to RGB
                elif batch.shape[1] != 3:
                    raise ValueError(f"Expected 1 or 3 channels, got {batch.shape[1]}")

                # Apply preprocessing (resize + normalize)
                batch = self.preprocess(batch)
            else:
                raise ValueError(f"Expected 4D tensor for images, got {batch.dim()}D")

            with torch.no_grad():
                self.logger.info(
                    f"AFTER TRANSFORM: Batch min {batch.min()}. Batch max {batch.max()}."
                )
                feat = self.feature_extractor(batch)
                # Global average pooling if spatial dimensions remain
                if feat.dim() > 2:
                    feat = F.adaptive_avg_pool2d(feat, (1, 1))
                feat = feat.view(feat.size(0), -1)  # Flatten
                features.append(feat.cpu())

        return torch.cat(features, dim=0)

    def cache_real_data(self, real_data, device):
        """
        Cache real data features for evaluation. Call this once before running evaluations.

        Args:
            real_data (torch.Tensor): Real training data tensor
            device: Device to use for computation
        """
        self.logger.info("Caching real data features...")
        self.device = device

        # Setup feature extractor if needed
        if self.feature_extractor is None and self.dataset_type == "image":
            self.setup_feature_extractor(device)

        # Move to device and extract features
        real_data = real_data.to(device)
        self.real_features_cache = self._extract_features(real_data).cpu()
        self.logger.info(
            f"Cached features for {len(self.real_features_cache)} real samples"
        )

    def _compute_wasserstein_distance(self, real_features, fake_features):
        """Compute 1-Wasserstein distance"""
        if real_features.shape[1] > 1:
            # For high-dimensional data, compute per-dimension and average
            distances = []
            for dim in range(
                min(real_features.shape[1], 10)
            ):  # Limit to first 10 dims for speed
                wd = wasserstein_distance(
                    real_features[:, dim].detach().cpu().numpy(),
                    fake_features[:, dim].detach().cpu().numpy(),
                )
                distances.append(wd)
            return np.mean(distances)
        else:
            return wasserstein_distance(
                real_features.flatten().detach().cpu().numpy(),
                fake_features.flatten().detach().cpu().numpy(),
            )

    def _compute_mmd(self, real_features, fake_features, kernel="rbf"):
        """Compute MMD using GPU operations"""
        X = real_features
        Y = fake_features

        if kernel == "rbf":
            # Use multiple bandwidths for better results
            bandwidths = [0.1, 1.0, 10.0]
            mmd_vals = []

            for bw in bandwidths:
                gamma = 1.0 / (2 * bw**2)

                # Compute kernel matrices on GPU
                XX = torch.cdist(X, X, p=2) ** 2
                YY = torch.cdist(Y, Y, p=2) ** 2
                XY = torch.cdist(X, Y, p=2) ** 2

                K_XX = torch.exp(-gamma * XX)
                K_YY = torch.exp(-gamma * YY)
                K_XY = torch.exp(-gamma * XY)

                mmd_val = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
                mmd_vals.append(mmd_val.item())

            return np.mean(mmd_vals)

        elif kernel == "polynomial":
            # Polynomial kernel: (X·Y + 1)^d
            degree = 2
            K_XX = (torch.matmul(X, X.t()) + 1) ** degree
            K_YY = (torch.matmul(Y, Y.t()) + 1) ** degree
            K_XY = (torch.matmul(X, Y.t()) + 1) ** degree

            mmd_val = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
            return mmd_val.item()

    def _compute_coverage_precision(self, real_features, fake_features, k=5):
        """Compute Coverage and Precision metrics using k-NN"""
        # Compute all pairwise distances on GPU
        real_real_dist = torch.cdist(real_features, real_features, p=2)
        fake_fake_dist = torch.cdist(fake_features, fake_features, p=2)
        real_fake_dist = torch.cdist(real_features, fake_features, p=2)

        # Get k-th nearest neighbor distances
        real_knn_dist, _ = torch.topk(real_real_dist, k + 1, dim=1, largest=False)
        real_radii = real_knn_dist[:, k]  # k-th nearest neighbor distance

        fake_knn_dist, _ = torch.topk(fake_fake_dist, k + 1, dim=1, largest=False)
        fake_radii = fake_knn_dist[:, k]

        # Coverage: fraction of real samples covered by fake samples
        fake_to_real_min_dist, _ = torch.min(real_fake_dist, dim=1)
        coverage = (fake_to_real_min_dist < real_radii).float().mean().item()

        # Precision: fraction of fake samples that are close to real samples
        real_to_fake_min_dist, _ = torch.min(real_fake_dist, dim=0)
        precision = (real_to_fake_min_dist < fake_radii).float().mean().item()

        return coverage, precision

    def _compute_js_divergence(self, real_features, fake_features, bins=50):
        """Compute Jensen-Shannon divergence between feature distributions"""
        eps = 1e-10
        js_divs = []

        n_dims = min(real_features.shape[1], 10)
        for dim in range(n_dims):
            real_vals = real_features[:, dim]
            fake_vals = fake_features[:, dim]

            min_val = torch.min(real_vals.min(), fake_vals.min())
            max_val = torch.max(real_vals.max(), fake_vals.max())
            if min_val == max_val:
                js_divs.append(torch.tensor(0.0, device=real_vals.device))
                continue

            real_hist = torch.histc(
                real_vals, bins=bins, min=min_val.item(), max=max_val.item()
            )
            fake_hist = torch.histc(
                fake_vals, bins=bins, min=min_val.item(), max=max_val.item()
            )

            real_hist = real_hist + eps
            fake_hist = fake_hist + eps

            real_hist /= real_hist.sum()
            fake_hist /= fake_hist.sum()
            m = 0.5 * (real_hist + fake_hist)

            js = (
                0.5
                * (
                    real_hist * torch.log2(real_hist / m)
                    + fake_hist * torch.log2(fake_hist / m)
                ).sum()
            )
            js_divs.append(js)

        return torch.stack(js_divs).mean().item()

    def _compute_energy_distance(self, real_features, fake_features, device):
        """Compute Energy Distance between two samples"""
        X = real_features.to(device)
        Y = fake_features.to(device)

        # Subsample for efficiency if datasets are large
        if len(X) > 500:
            idx_x = torch.randperm(len(X), device=device)[:500]
            X = X[idx_x]
        if len(Y) > 500:
            idx_y = torch.randperm(len(Y), device=device)[:500]
            Y = Y[idx_y]

        # Compute pairwise distances on GPU
        dXY = torch.cdist(X, Y, p=2).mean()
        dXX = torch.cdist(X, X, p=2).mean()
        dYY = torch.cdist(Y, Y, p=2).mean()

        energy_distance = 2 * dXY - dXX - dYY
        return energy_distance.item()

    def _compute_density_consistency(self, real_features, fake_features, k=5):
        """Measure how well fake samples match real data density using k-NN density estimation"""
        # Compute distances for density estimation
        real_real_dist = torch.cdist(real_features, real_features, p=2)
        real_fake_dist = torch.cdist(real_features, fake_features, p=2)

        # Get k-th nearest neighbor distances
        real_knn_dist, _ = torch.topk(real_real_dist, k + 1, dim=1, largest=False)
        real_densities = 1.0 / (real_knn_dist[:, k] + 1e-10)

        fake_knn_dist, _ = torch.topk(real_fake_dist, k + 1, dim=0, largest=False)
        fake_densities = 1.0 / (fake_knn_dist[k, :] + 1e-10)

        # Convert to CPU for statistical tests
        real_densities_cpu = real_densities.detach().cpu().numpy()
        fake_densities_cpu = fake_densities.detach().cpu().numpy()

        # KS test
        ks_stat, _ = kstest(fake_densities_cpu, real_densities_cpu)

        # Density ratio
        density_ratio = torch.log(
            fake_densities.mean() / (real_densities.mean() + 1e-10)
        ).item()

        return {"density_ks_stat": ks_stat, "log_density_ratio": density_ratio}

    def _compute_mode_collapse_metrics(self, fake_features, k=10):
        """Detect mode collapse using intra-cluster distance and nearest neighbor analysis"""
        if len(fake_features) < k:
            return {"mode_collapse_score": 0.0, "duplicate_ratio": 0.0}

        # Compute pairwise distances
        distances = torch.cdist(fake_features, fake_features, p=2)

        # Get k-th nearest neighbor distances
        knn_distances, _ = torch.topk(distances, k + 1, dim=1, largest=False)

        # Mode collapse score: average k-th nearest neighbor distance
        mode_collapse_score = knn_distances[:, k].mean().item()

        # Duplicate detection using 1st nearest neighbor
        nn_distances = knn_distances[:, 1]  # 1st nearest neighbor
        duplicate_threshold = torch.quantile(nn_distances, 0.05)  # 5th percentile
        duplicate_ratio = (nn_distances < duplicate_threshold).float().mean().item()

        return {
            "mode_collapse_score": mode_collapse_score,
            "duplicate_ratio": duplicate_ratio,
        }

    def _compute_diversity_metrics(self, fake_features, device):
        """Compute diversity metrics for generated samples"""
        fake_features = fake_features.to(device)
        # Subsample for efficiency
        if len(fake_features) > 500:
            idx = torch.randperm(len(fake_features), device=device)[:500]
            fake_subset = fake_features[idx]
        else:
            fake_subset = fake_features

        # Compute pairwise distances
        distances = torch.cdist(fake_subset, fake_subset, p=2).to(device)

        # Remove diagonal (self-distances)
        mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=device)
        distances_flat = distances[mask]

        # Diversity metrics
        mean_pairwise_distance = distances_flat.mean().item()
        min_pairwise_distance = distances_flat.min().item()
        std_pairwise_distance = distances_flat.std().item()

        self.logger.info(
            f"Feature min/max:, {fake_subset.min().item()}, {fake_subset.max().item()}"
        )
        self.logger.info(f"Any NaNs?, {torch.isnan(fake_subset).any().item()}")
        self.logger.info(f"Any Infs?, {torch.isinf(fake_subset).any().item()}")

        # Distance entropy (convert to CPU for histogram)
        hist = torch.histc(
            distances_flat.cpu(),
            bins=50,
            min=distances_flat.min().item(),
            max=distances_flat.max().item(),
        )
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        distance_entropy = entropy(hist.detach().cpu().numpy(), base=2)

        return {
            "mean_pairwise_distance": mean_pairwise_distance,
            "min_pairwise_distance": min_pairwise_distance,
            "std_pairwise_distance": std_pairwise_distance,
            "distance_entropy": distance_entropy,
        }

    def _compute_fid(self, real_features, fake_features):
        """
        Compute Fréchet Inception Distance (FID) using GPU operations.
        FID measures the distance between two multivariate Gaussians.
        """
        # Ensure we have enough samples for stable statistics
        if len(real_features) < 10 or len(fake_features) < 10:
            return float("inf")

        # Compute means
        mu_real = torch.mean(real_features, dim=0)
        mu_fake = torch.mean(fake_features, dim=0)

        # Compute covariance matrices
        real_centered = real_features - mu_real
        fake_centered = fake_features - mu_fake

        sigma_real = torch.matmul(real_centered.t(), real_centered) / (
            len(real_features) - 1
        )
        sigma_fake = torch.matmul(fake_centered.t(), fake_centered) / (
            len(fake_features) - 1
        )

        # Add small diagonal for numerical stability
        eps = 1e-6
        sigma_real += eps * torch.eye(sigma_real.shape[0], device=real_features.device)
        sigma_fake += eps * torch.eye(sigma_fake.shape[0], device=fake_features.device)

        # Compute mean difference squared
        diff = mu_real - mu_fake
        mean_diff_sq = torch.dot(diff, diff)

        # Compute trace of covariances
        tr_sigma_real = torch.trace(sigma_real)
        tr_sigma_fake = torch.trace(sigma_fake)

        # Compute sqrt(sigma_real @ sigma_fake) using eigendecomposition
        # This is more stable than direct matrix square root
        try:
            # Compute sigma_real^(1/2) @ sigma_fake @ sigma_real^(1/2)
            eigenvals_real, eigenvecs_real = torch.linalg.eigh(sigma_real)
            eigenvals_real = torch.clamp(eigenvals_real, min=eps)  # Ensure positive

            sqrt_sigma_real = (
                eigenvecs_real
                @ torch.diag(torch.sqrt(eigenvals_real))
                @ eigenvecs_real.t()
            )

            # Compute sqrt_sigma_real @ sigma_fake @ sqrt_sigma_real
            middle_matrix = sqrt_sigma_real @ sigma_fake @ sqrt_sigma_real
            eigenvals_middle, _ = torch.linalg.eigh(middle_matrix)
            eigenvals_middle = torch.clamp(
                eigenvals_middle, min=0
            )  # Ensure non-negative

            tr_sqrt_product = torch.sum(torch.sqrt(eigenvals_middle))

        except Exception as e:
            # Fallback to approximation if eigendecomposition fails
            self.logger.warning(f"FID computation failed with eigendecomposition: {e}")
            # Use Frobenius norm approximation: ||A||_F ≈ sqrt(tr(A^T A))
            product_approx = torch.matmul(sigma_real, sigma_fake)
            tr_sqrt_product = torch.sqrt(torch.trace(product_approx) + eps)

        # Compute FID
        fid = mean_diff_sq + tr_sigma_real + tr_sigma_fake - 2 * tr_sqrt_product

        return fid.item()

    def is_cached(self):
        """Check if real data has been cached"""
        return self.real_features_cache is not None

    def evaluate(self, generated_samples):
        """
        Evaluate generated samples against cached real data.

        Args:
            generated_samples (torch.Tensor): Generated samples tensor

        Returns:
            dict: Dictionary of computed metrics

        Raises:
            ValueError: If real data hasn't been cached yet
        """
        if not self.is_cached():
            raise ValueError("Real data not cached. Call cache_real_data() first.")

        if self.device is None:
            raise ValueError("Device not set. Call cache_real_data() first.")

        # Extract features from generated samples
        generated_samples = generated_samples.to(self.device)
        fake_features = self._extract_features(generated_samples).cpu()

        metrics = {}

        self.logger.debug("Computing metrics...")

        # Compute FID
        if self.compute_fid:
            fid = self._compute_fid(self.real_features_cache, fake_features)
            metrics["fid"] = fid
            self.logger.debug("Computed FID.")

        # Compute Wasserstein distance
        if self.compute_wasserstein:
            wd = self._compute_wasserstein_distance(
                self.real_features_cache, fake_features
            )
            metrics["wasserstein_distance"] = wd
            self.logger.debug("Computed Wasserstein distance.")

        # Compute MMD
        if self.compute_mmd:
            mmd = self._compute_mmd(
                self.real_features_cache, fake_features, self.mmd_kernel
            )
            metrics["mmd"] = mmd
            self.logger.debug("Computed MMD.")

        # Compute Coverage and Precision
        if self.compute_coverage_precision:
            coverage, precision = self._compute_coverage_precision(
                self.real_features_cache, fake_features, self.k_nearest
            )
            metrics["coverage"] = coverage
            metrics["precision"] = precision
            self.logger.debug("Computed coverage and precision.")

        # Compute Jensen-Shannon Divergence
        if self.compute_js_divergence:
            js_div = self._compute_js_divergence(
                self.real_features_cache, fake_features
            )
            metrics["js_divergence"] = js_div
            self.logger.debug("Computed JS divergence.")

        # Compute Energy Distance
        if self.compute_energy_distance:
            energy_dist = self._compute_energy_distance(
                self.real_features_cache, fake_features, self.device
            )
            metrics["energy_distance"] = energy_dist
            self.logger.debug("Computed energy distance.")

        # Compute Density Consistency
        if self.compute_density_consistency:
            density_metrics = self._compute_density_consistency(
                self.real_features_cache, fake_features
            )
            metrics.update(density_metrics)
            self.logger.debug("Computed density consistency.")

        # Compute Mode Collapse Metrics
        if self.compute_mode_collapse:
            mode_metrics = self._compute_mode_collapse_metrics(fake_features)
            metrics.update(mode_metrics)
            self.logger.debug("Computed mode collapse metrics.")

        # Compute Diversity Metrics
        if self.compute_diversity_metrics:
            diversity_metrics = self._compute_diversity_metrics(
                fake_features, self.device
            )
            metrics.update(diversity_metrics)
            self.logger.debug("Computed diversity metrics.")

        return metrics

    def clear_cache(self):
        """Clear cached real data features"""
        self.real_features_cache = None
        self.device = None
        self.logger.info("Cleared real data cache.")
