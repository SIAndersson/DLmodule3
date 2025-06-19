import torch
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance
import pytorch_lightning as pl
from torchvision.utils import save_image
from pathlib import Path
from pytorch_lightning.callbacks import Callback


class EvaluateSamplesCallback(Callback):
    def __init__(self, num_samples=500, save_dir="eval_outputs", model_type="vector_field"):
        self.num_samples = num_samples
        self.save_dir = Path(save_dir)
        self.fid = FrechetInceptionDistance(normalize=True)
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
                if hasattr(model, 'sample'):
                    if self.model_type=="vector_field":
                        x_hat = model.sample(x.shape[0]) # Batch size
                    elif self.model_type=="diffusion":
                        x_hat = model.sample(x.shape, x.device)
                    else:
                        raise ValueError(f"Unsupported model type: {self.model_type}")
                else:
                    print("Model lacks sample() method")
                    break

                # Store real and fake
                real_imgs.append(self._denormalize(x))
                fake_imgs.append(self._denormalize(x_hat))

        real_imgs = torch.cat(real_imgs, dim=0)[:self.num_samples].to(x.device)
        fake_imgs = torch.cat(fake_imgs, dim=0)[:self.num_samples].to(x.device)
        return real_imgs, fake_imgs

    def on_train_epoch_end(self, trainer, pl_module):
        try:
            model_head = pl_module.hparams.model_cfg.model_type
        except AttributeError:
            print("model_type not found in hparams — skipping evaluation.")
            return

        if model_head != "CNN":
            print("Skipping evaluation (non-CNN model)")
            return

        dataloader = trainer.datamodule.train_dataloader()
        real_imgs, fake_imgs = self._generate_and_collect(pl_module, dataloader)

        self.fid.reset()
        self.fid.update(real_imgs.to(real_imgs.device), real=True)
        self.fid.update(fake_imgs.to(fake_imgs.device), real=False)
        fid_score = self.fid.compute().item()

        # Save sample grid
        grid_path = self.save_dir / f"{self.model_type}_generated_samples_epoch{trainer.current_epoch}.png"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_image(fake_imgs[:16], grid_path, nrow=4)

        pl_module.log("fid", fid_score, on_epoch=True, prog_bar=True)


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
