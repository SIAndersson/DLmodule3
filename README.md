# Articles to read

## Diffusion

- [Diffusion and flow matching equivalent](https://diffusionflow.github.io/)
- [DDIM sampler implementation](https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html)
- [DDIM paper](https://arxiv.org/pdf/2010.02502)
- [Adding noise to data for diffusion performance image gen](https://arxiv.org/pdf/2301.11706v3)
- [Diffusion model introduction](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

## Flow matching

- [First paper on flow matching](https://arxiv.org/abs/2210.02747)
- [Introduction to flow matching](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)
- [Dirichlet flow matching for DNA sequence design](https://arxiv.org/pdf/2402.05841)

# Initial layout

Generative models:

- Diffusion (Lo)
  - Linear schedule diffusion OR classifier/classifier-free guidance
  - DDPM with Cosine schedule diffusion
  - DDIM (Denoising Diffusion Implicit Models)
- Flow matching (Seb)
  - Standard flow matching
  - Dirichlet flow matching
  - Classifier/classifier-free guidance

Datasets:

- Two moons (example?)
- FFHQ
- LSUN Tower

Generative model aspects:

- Quality of generated samples
- Training stability and convergence

Evaluation metrics:

- Loss (?)
- Frechet Inception Distance (FID) (measures the similarity between the distributions of the generated samples and the real data in the feature space of a pre-trained classifier)
- Inception Score (IS) (classify the generated samples and compute the entropy of the predicted class probabilities)
- Perceptual Path Length (PPL) (calculated by measuring the average perceptual distance between pairs of generated samples that are interpolated in the latent space)
- Precision, Recall, F1-score (computed using nearest-neighbor matching in the feature space of a pre-trained classifier)
- Log-Likelihood (measures the probability of the real data given the model)
