import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from pytorch_lightning.callbacks import Callback
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image

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


# This is SO EXTREMELY SLOW
class EvaluateSamplesCallback(Callback):
    def __init__(
        self, num_samples=500, save_dir="eval_outputs", model_type="vector_field"
    ):
        self.num_samples = num_samples
        self.save_dir = Path(save_dir)
        self.fid = FrechetInceptionDistance(
            normalize=True, feature_extractor_weights_path=DATASET_CACHE
        )
        self.model_type = model_type

    def _denormalize(self, img):
        # [-1, 1] -> [0, 1]
        return (img + 1) / 2.0

    def _generate_and_collect(self, model, dataloader):
        real_imgs = []
        fake_imgs = []

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch

                # Break if we have enough
                if len(real_imgs) * x.size(0) >= self.num_samples:
                    break

                # Generate samples
                if hasattr(model, "sample"):
                    if self.model_type == "vector_field":
                        x_hat = model.sample(x.shape[0], model.device)  # Batch size
                    elif self.model_type == "diffusion":
                        x_hat = model.sample(x.shape, model.device)
                    else:
                        raise ValueError(f"Unsupported model type: {self.model_type}")
                else:
                    print("Model lacks sample() method")
                    break

                # Store real and fake
                real_imgs.append(self._denormalize(x))
                fake_imgs.append(self._denormalize(x_hat))

        real_imgs = torch.cat(real_imgs, dim=0)[: self.num_samples].to(x.device)
        fake_imgs = torch.cat(fake_imgs, dim=0)[: self.num_samples].to(x.device)
        return real_imgs, fake_imgs

    def on_train_epoch_end(self, trainer, pl_module):
        try:
            model_head = pl_module.hparams.model_cfg.model_type
        except AttributeError:
            print("model_type not found in hparams — skipping evaluation.")
            return

        if model_head.upper() != "CNN":
            print("Skipping evaluation (non-CNN model)")
            return

        dataloader = trainer.train_dataloader
        real_imgs, fake_imgs = self._generate_and_collect(pl_module, dataloader)

        self.fid.reset()
        self.fid.update(real_imgs.to(real_imgs.device), real=True)
        self.fid.update(fake_imgs.to(fake_imgs.device), real=False)
        fid_score = self.fid.compute().item()

        # Save sample grid
        grid_path = (
            self.save_dir
            / f"{self.model_type}_generated_samples_epoch{trainer.current_epoch}.png"
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_image(fake_imgs[:16], grid_path, nrow=4)

        pl_module.log("fid", fid_score, on_epoch=True, prog_bar=True)


# The cooler Daniel (the faster callback)
class FastEvaluationCallback(Callback):
    """
    Fast evaluation callback for generative models.
    Computes lightweight metrics during training without slowing down significantly.
    """

    def __init__(
        self,
        real_data: torch.Tensor,
        dataset_type: str = "2d",  # "2d" or "image"
        model_type: str = "vector_field",  # "vector_field" or "diffusion"
        eval_samples: int = 1000,  # Number of samples to generate for evaluation
        eval_frequency: int = 10,  # Evaluate every N epochs
        k_nearest: int = 5,  # For coverage/precision metrics
        # New parameters for optimization
        use_lightweight_metrics: bool = True,  # Use FD instead of KID/IS
        feature_dim: int = 512,  # Dimension for lightweight feature extractor
        batch_size: int = 64,  # Batch size for feature extraction
        max_eval_samples: int = 500,  # Cap samples for image metrics
    ):
        super().__init__()
        self.real_data = real_data
        self.dataset_type = dataset_type.lower()
        self.eval_samples = eval_samples
        self.eval_frequency = eval_frequency
        self.k_nearest = k_nearest
        self.model_type = model_type.lower()
        self.use_lightweight_metrics = use_lightweight_metrics
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.max_eval_samples = max_eval_samples

        # Precompute real data statistics
        self.real_data_cpu = real_data.cpu().numpy()

        if self.dataset_type == "image":
            if self.use_lightweight_metrics:
                # Use a lightweight CNN feature extractor instead of Inception
                self.feature_extractor = self._create_lightweight_extractor()
            else:
                # Use smaller, more efficient model than full Inception
                self.feature_extractor = self._create_efficient_inception()

            self.feature_extractor.eval()

            # Precompute real data features with reduced samples
            real_subset_size = min(len(real_data), self.max_eval_samples)
            real_subset_indices = torch.randperm(len(real_data))[:real_subset_size]
            real_subset = real_data[real_subset_indices]
            self._precompute_real_features(real_subset)

        # Store metrics history
        self.metrics_history = {
            "epoch": [],
            "wasserstein_dist": [],
            "coverage": [],
            "precision": [],
            "mmd": [],
        }

        if self.dataset_type == "image":
            if self.use_lightweight_metrics:
                self.metrics_history.update(
                    {"frechet_distance": [], "lpips_diversity": []}
                )
            else:
                self.metrics_history.update({"inception_score": [], "kid_score": []})

    def _create_lightweight_extractor(self):
        """Create a lightweight CNN feature extractor"""

        class LightweightExtractor(torch.nn.Module):
            def __init__(self, feature_dim=512):
                super().__init__()
                # Simple but effective feature extractor
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 3, 2, 1),  # 264->132
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(64, 128, 3, 2, 1),  # 132->66
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(128, 256, 3, 2, 1),  # 66->33
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(256, 512, 3, 2, 1),  # 33->17
                    torch.nn.ReLU(inplace=True),
                    torch.nn.AdaptiveAvgPool2d((4, 4)),  # 17->4
                    torch.nn.Flatten(),
                    torch.nn.Linear(512 * 16, feature_dim),
                    torch.nn.ReLU(inplace=True),
                )

            def forward(self, x):
                # Normalize input to [0, 1] if needed
                if x.min() < 0:
                    x = (x + 1) / 2
                return self.features(x)

        return LightweightExtractor(self.feature_dim)

    def _create_efficient_inception(self):
        """Create a more efficient version of Inception"""
        # Use MobileNetV2 instead of Inception - much faster
        from torchvision.models import mobilenet_v2

        model = mobilenet_v2(pretrained=True)
        # Remove classifier, keep features
        model.classifier = torch.nn.Identity()
        return model

    def _precompute_real_features(self, real_data):
        """Precompute features for real data to speed up evaluation"""
        device = real_data.device
        self.feature_extractor = self.feature_extractor.to(device)

        features = []
        with torch.no_grad():
            for i in range(0, len(real_data), self.batch_size):
                batch = real_data[i : i + self.batch_size]

                if self.use_lightweight_metrics:
                    # No resizing needed for lightweight extractor
                    feat = self.feature_extractor(batch)
                else:
                    # Only resize if using MobileNet (expects 224x224)
                    if batch.size(-1) != 224:
                        batch = F.interpolate(
                            batch, size=(224, 224), mode="bilinear", align_corners=False
                        )
                    feat = self.feature_extractor(batch)

                features.append(feat.cpu())

        self.real_features = torch.cat(features, dim=0)

    def on_train_epoch_end(self, trainer, pl_module):
        """Run evaluation at the end of validation epoch"""
        if trainer.current_epoch % self.eval_frequency != 0:
            return

        # Generate samples (with reduced count for images)
        pl_module.eval()
        device = next(pl_module.parameters()).device

        eval_samples = self.eval_samples
        if self.dataset_type == "image":
            eval_samples = min(self.eval_samples, self.max_eval_samples)

        with torch.no_grad():
            if self.dataset_type == "2d":
                if self.model_type == "vector_field":
                    generated_samples = pl_module.sample(eval_samples, device)
                elif self.model_type == "diffusion":
                    generated_samples = pl_module.sample((eval_samples, 2), device)
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")
            else:  # image
                if self.model_type == "vector_field":
                    generated_samples = pl_module.fast_sample(eval_samples, device)
                elif self.model_type == "diffusion":
                    generated_samples = pl_module.ddim_sample(
                        (eval_samples, 3, 264, 264), device
                    )
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")

        # Compute metrics
        metrics = self._compute_metrics(generated_samples, device)

        # Log metrics
        for metric_name, value in metrics.items():
            pl_module.log(
                f"{metric_name}", value, on_epoch=True, prog_bar=True, sync_dist=True
            )

        # Store in history
        self.metrics_history["epoch"].append(trainer.current_epoch)
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append(value)

    def _compute_metrics(self, generated_samples, device):
        """Compute all relevant metrics"""
        metrics = {}

        if self.dataset_type == "2d":
            metrics.update(self._compute_2d_metrics(generated_samples))
        else:
            metrics.update(self._compute_image_metrics(generated_samples, device))

        return metrics

    def _compute_2d_metrics(self, generated_samples):
        """Compute metrics for 2D datasets"""
        gen_data = generated_samples.cpu().numpy()

        metrics = {}

        # 1. Wasserstein Distance (1D projections)
        w_distances = []
        for dim in range(2):
            wd = wasserstein_distance(self.real_data_cpu[:, dim], gen_data[:, dim])
            w_distances.append(wd)
        metrics["wasserstein_dist"] = np.mean(w_distances)

        # 2. Coverage and Precision
        coverage, precision = self._compute_coverage_precision(
            self.real_data_cpu, gen_data
        )
        metrics["coverage"] = coverage
        metrics["precision"] = precision

        # 3. MMD with RBF kernel
        mmd_score = self._compute_mmd_rbf(self.real_data_cpu, gen_data)
        metrics["mmd"] = mmd_score

        return metrics

    def _compute_image_metrics(self, generated_samples, device):
        """Compute metrics for image datasets"""
        metrics = {}

        if self.use_lightweight_metrics:
            # Use faster, lighter metrics
            fd_score = self._compute_frechet_distance(generated_samples, device)
            metrics["frechet_distance"] = fd_score

            # Simple diversity metric
            diversity_score = self._compute_diversity(generated_samples)
            metrics["lpips_diversity"] = diversity_score
        else:
            # Use traditional but optimized metrics
            is_score = self._compute_inception_score_fast(generated_samples, device)
            metrics["inception_score"] = is_score

            kid_score = self._compute_kid_fast(generated_samples, device)
            metrics["kid_score"] = kid_score

        return metrics

    def _compute_frechet_distance(self, generated_samples, device):
        """Compute Frechet Distance using lightweight features"""
        self.feature_extractor = self.feature_extractor.to(device)

        with torch.no_grad():
            gen_features = []
            for i in range(0, len(generated_samples), self.batch_size):
                batch = generated_samples[i : i + self.batch_size]
                feat = self.feature_extractor(batch)
                gen_features.append(feat)

            gen_features = torch.cat(gen_features, dim=0).to(device)

        # Compute Frechet distance
        real_features = self.real_features

        # Compute means and covariances
        mu_real, sigma_real = (
            torch.mean(real_features, dim=0).to(device),
            cov(real_features, rowvar=False).to(device),
        )
        mu_gen, sigma_gen = (
            torch.mean(gen_features, dim=0).to(device),
            cov(gen_features, rowvar=False).to(device),
        )

        # Compute Frechet distance
        fd = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
        return fd

    def _compute_diversity(self, generated_samples):
        """Compute a simple diversity metric based on pairwise distances"""
        # Subsample for efficiency
        n_samples = min(100, len(generated_samples))
        indices = torch.randperm(len(generated_samples))[:n_samples]
        samples = generated_samples[indices].flatten(1)  # Flatten spatial dims

        # Compute pairwise L2 distances
        pairwise_dists = torch.cdist(samples, samples, p=2.0)

        # Return mean pairwise distance (higher = more diverse)
        mask = torch.triu(torch.ones_like(pairwise_dists, dtype=torch.bool), diagonal=1)
        return torch.mean(pairwise_dists[mask])

    def _compute_inception_score_fast(self, images, device, splits=5):
        """Faster Inception Score with reduced splits and batch processing"""
        self.feature_extractor = self.feature_extractor.to(device)

        with torch.no_grad():
            # Process in larger batches
            preds = []
            for i in range(0, len(images), self.batch_size):
                batch = images[i : i + self.batch_size]
                if batch.size(-1) != 224:
                    batch = F.interpolate(
                        batch, size=(224, 224), mode="bilinear", align_corners=False
                    )

                # Normalize to [0, 1]
                if batch.min() < 0:
                    batch = (batch + 1) / 2

                pred = F.softmax(self.feature_extractor(batch), dim=1)
                preds.append(pred.cpu())

            preds = torch.cat(preds, dim=0)

        # Compute IS with fewer splits
        scores = []
        for i in range(splits):
            part = preds[i * len(preds) // splits : (i + 1) * len(preds) // splits]

            # Compute KL divergence
            mean_part = torch.mean(part, dim=0, keepdim=True)
            kl_div = part * (torch.log(part + 1e-10) - torch.log(mean_part + 1e-10))
            kl_div = torch.mean(torch.sum(kl_div, dim=1))

            scores.append(torch.exp(kl_div))

        return torch.mean(torch.tensor(scores))

    def _compute_kid_fast(self, generated_samples, device, subset_size=300):
        """Faster KID with smaller subset"""
        self.feature_extractor = self.feature_extractor.to(device)

        with torch.no_grad():
            gen_features = []
            for i in range(0, len(generated_samples), self.batch_size):
                batch = generated_samples[i : i + self.batch_size]
                if batch.size(-1) != 224:
                    batch = F.interpolate(
                        batch, size=(224, 224), mode="bilinear", align_corners=False
                    )

                if batch.min() < 0:
                    batch = (batch + 1) / 2

                feat = self.feature_extractor(batch)
                gen_features.append(feat.cpu())

            gen_features = torch.cat(gen_features, dim=0)

        # Use smaller subsets
        n_samples = min(subset_size, len(self.real_features), len(gen_features))

        real_indices = torch.randperm(len(self.real_features))[:n_samples]
        gen_indices = torch.randperm(len(gen_features))[:n_samples]

        real_subset = self.real_features[real_indices]
        gen_subset = gen_features[gen_indices]

        # Compute polynomial kernel MMD
        kid_score = self._polynomial_mmd(real_subset.numpy(), gen_subset.numpy())
        return kid_score

    def _compute_coverage_precision(self, real_data, gen_data, k=5):
        """Compute coverage and precision metrics."""
        # Use subsets for efficiency
        max_points = 500
        if len(real_data) > max_points:
            real_indices = np.random.choice(len(real_data), max_points, replace=False)
            real_data = real_data[real_indices]
        if len(gen_data) > max_points:
            gen_indices = np.random.choice(len(gen_data), max_points, replace=False)
            gen_data = gen_data[gen_indices]

        # Fit nearest neighbors
        nbrs_real = NearestNeighbors(n_neighbors=min(k, len(real_data))).fit(real_data)
        nbrs_gen = NearestNeighbors(n_neighbors=min(k, len(gen_data))).fit(gen_data)

        # Coverage
        distances_real_to_gen, _ = nbrs_gen.kneighbors(real_data)
        threshold_real = np.percentile(distances_real_to_gen[:, 0], 95)
        coverage = np.mean(distances_real_to_gen[:, 0] < threshold_real)

        # Precision
        distances_gen_to_real, _ = nbrs_real.kneighbors(gen_data)
        threshold_gen = np.percentile(distances_gen_to_real[:, 0], 95)
        precision = np.mean(distances_gen_to_real[:, 0] < threshold_gen)

        return coverage, precision

    def _compute_mmd_rbf(self, X, Y, gamma=1.0):
        """Compute Maximum Mean Discrepancy with RBF kernel"""
        # Use smaller subsets
        n_samples = min(300, len(X), len(Y))
        X_sub = X[np.random.choice(len(X), n_samples, replace=False)]
        Y_sub = Y[np.random.choice(len(Y), n_samples, replace=False)]

        # Compute pairwise distances
        XX = cdist(X_sub, X_sub, metric="euclidean")
        YY = cdist(Y_sub, Y_sub, metric="euclidean")
        XY = cdist(X_sub, Y_sub, metric="euclidean")

        # RBF kernel
        K_XX = np.exp(-gamma * XX**2)
        K_YY = np.exp(-gamma * YY**2)
        K_XY = np.exp(-gamma * XY**2)

        mmd_sq = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        return np.sqrt(max(mmd_sq, 0))

    def _polynomial_mmd(self, X, Y, degree=3, gamma=None, coef0=1):
        """Compute MMD with polynomial kernel"""
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        K_XX = (gamma * np.dot(X, X.T) + coef0) ** degree
        K_YY = (gamma * np.dot(Y, Y.T) + coef0) ** degree
        K_XY = (gamma * np.dot(X, Y.T) + coef0) ** degree

        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)

        n, m = X.shape[0], Y.shape[0]
        mmd_sq = (
            K_XX.sum() / (n * (n - 1))
            + K_YY.sum() / (m * (m - 1))
            - 2 * K_XY.sum() / (n * m)
        )

        return np.sqrt(max(mmd_sq, 0))

    def get_metrics_history(self):
        """Return the complete metrics history"""
        return self.metrics_history


class MetricTracker(Callback):
    def __init__(self):
        self.train_losses = []
        self.fid_scores = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        fid = trainer.callback_metrics.get("fid")
        if loss is not None:
            self.train_losses.append(loss.item())
        if fid is not None:
            self.fid_scores.append(fid.item())


def create_evaluation_callback(cfg, data, model_type):
    """Create the appropriate evaluation callback based on dataset type"""

    if cfg.main.dataset.lower() in ["two_moons", "2d_gaussians"]:
        return FastEvaluationCallback(
            real_data=data[:1100],
            model_type=model_type,
            dataset_type="2d",
            eval_samples=1000,
            eval_frequency=5,  # Evaluate every 5 epochs
            k_nearest=5,
        )
    else:
        return FastEvaluationCallback(
            real_data=data[:600],
            model_type=model_type,
            dataset_type="image",
            eval_samples=500,  # Fewer samples for images to keep it fast
            eval_frequency=10,
            k_nearest=5,
        )
