# FACE3 correctness audit that motivated FACE4

FACE4 preserves FACE3 as a separate repository. This document records the
bugs found by tracing FACE3 source code against its pushed histories and
public stock replays.

## Critical findings

1. **The optimization editor and stock editor were different functions.**
   FACE3 multiplied the image-conditioning VAE latent by
   `vae.config.scaling_factor` only when gradients/checkpointing were active.
   InstructPix2Pix's stock `prepare_image_latents` path does not apply this
   scaling to the conditioning latent.

2. **The old parity test skipped the broken branch.** It called the wrapper
   inside `torch.no_grad()`. That selected the correct stock-like helper path,
   while optimization selected the incorrect gradient-only path.

3. **The old checkpoint closure captured the denoising timestep late.** During
   backward recomputation, checkpointed UNet calls could use the final loop
   timestep instead of each step's own timestep.

4. **ArcFace's residual downsample topology was reversed.** FACE3 constructed
   BatchNorm then Conv, while the InsightFace iResNet uses Conv then BatchNorm.
   Pushed summaries show 20 missing downsample tensors and a 0.974 state-dict
   match ratio; those objective layers were random.

5. **Rows mixed different optimizer states.** Z/images/flows were computed
   before `optimizer.step`, while parameter diagnostics and theta snapshots
   were captured after it. Final images and final-row geometry metrics were
   also offset by one update.

6. **Continuous optimization did not include the PNG roundtrip.** Public stock
   replay consumed an 8-bit PIL image, while the optimization editor consumed
   an unquantized float tensor. FACE4 uses an exact 8-bit forward value with a
   documented straight-through gradient for rounding.

7. **`Z` meant different quantities in different files.** Histories used the
   reconstructed gradient-path Z, while summaries overwrote `final_Z` and
   `best_Z` with a limited stock replay. `best_iter_by_Z` still referred to the
   reconstructed-path candidate.

8. **DeepFace evaluation contained an undefined name.** It imported
   `DeepFace` but called `Deepface3.verify`, so every non-skipped panel failed.

9. **The per-iteration SSIM was a global-statistics approximation.** FACE4
   replaces it with an 11x11 Gaussian-window SSIM calculation.

## Evidence from the pushed runs

The same initial perturbation was evaluated at FACE3 history rows 0 and 1,
before any optimizer update could change it. For `face_002 + sunglasses`, both
rows reported the same input SSIM (`0.999639213`) and maximum displacement
(`0.825897872 px`), but Z changed from `0.998624265` to `0.937189937`. The only
change was `no_grad` versus the defective grad-only latent branch.

Across the four run-6 cases, this branch discontinuity accounted for roughly
83.5% to 92.2% of the total apparent Z reduction. Gradient-best edited outputs
had only about 0.23 to 0.35 SSIM against stock replay, while no-grad/final
outputs were near stock parity.

## FACE4 policy

FACE4 does not silently accept another approximation. It executes the exact
installed Diffusers pipeline body after unwrapping only the outer no-grad
decorator. Before optimization, and periodically during it, FACE4 compares:

- gradient-enabled exact output;
- no-grad exact output;
- decorated stock pipeline output on the same quantized tensor;
- normal PIL-input/PIL-output stock replay;
- their ArcFace Z values.

A run fails instead of producing a falling Z graph if these paths exceed the
configured parity tolerances. ArcFace checkpoint loading must also be exact.

Primary implementation references:

- Hugging Face Diffusers InstructPix2Pix pipeline:
  <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py>
- InsightFace iResNet backbone:
  <https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/iresnet.py>
