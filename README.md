# FACE4

FACE4 is the correctness-first successor to FACE3. It optimizes geometric
perturbations through an edited-output identity objective:

```text
clean_edit      = InstructPix2Pix(original, prompt, fixed seed/settings)
perturbed_input = quantize8(T_theta(original))
perturbed_edit  = InstructPix2Pix(perturbed_input, prompt, same seed/settings)

Z    = cosine_similarity(ArcFace(clean_edit), ArcFace(perturbed_edit))
loss = Z
```

Adam minimizes `loss = Z`. InstructPix2Pix and ArcFace weights are frozen;
only enabled perturbation parameters are updated.

## What FACE4 corrects

FACE3 used a hand-reconstructed differentiable denoising loop. Its
gradient-only branch incorrectly scaled the InstructPix2Pix image-conditioning
latent, while its no-grad/stock branch did not. It also late-bound checkpointed
timesteps and loaded ArcFace into a mismatched residual downsample topology.
Those bugs made the falling optimization Z different from stock replay Z.

FACE4 changes the execution contract:

- the gradient path runs the exact installed Diffusers pipeline body by
  unwrapping only its outer `torch.no_grad` decorator;
- no hand-reconstructed denoising loop is used;
- Diffusers' own preprocessing, conditioning-latent logic, guidance, scheduler,
  denoising, and decoder are shared by optimization and stock replay;
- the forward input and edited output are exactly 8-bit quantized, matching the
  saved PNG artifacts; rounding uses a straight-through estimator only for the
  backward derivative;
- the ArcFace iResNet topology matches InsightFace Conv-to-BN downsample blocks,
  and checkpoint loading must be exact;
- fp16 backward uses adaptive loss scaling and unscales geometry gradients
  before Adam. If a state overflows, FACE4 retries that same state at a lower
  scale before any optimizer update; this prevents underflow/overflow without
  changing `loss = Z`;
- every logged Z, image, flow, parameter diagnostic, and best iteration refers
  to one consistent pre-update geometry state;
- stock Z is checked at initialization, periodically, and at final/best replay.

The complete FACE3 audit is in
[`docs/FACE3_CORRECTNESS_AUDIT.md`](docs/FACE3_CORRECTNESS_AUDIT.md).

## Mandatory parity gate

Before a case is optimized, FACE4 compares all of the following on the same
canonical 8-bit input, prompt, seed, and settings:

1. gradient-enabled exact Diffusers forward;
2. no-grad exact Diffusers forward;
3. decorated stock pipeline with tensor input;
4. normal stock pipeline with PIL input;
5. ArcFace Z from each output;
6. a real backward pass to the editor input.
7. checkpoint-enabled versus checkpoint-disabled input gradients in the
   standalone correctness command.

The run fails instead of writing a misleading curve if parity exceeds the
configured thresholds. Exact tensor execution and ordinary PIL replay use the
separate tolerances documented under Output semantics below.

## Geometry

The default JSON enables TPS, fixed-topology Delaunay/piecewise affine, and
rolling-shutter coordinate warps together. Additional independently toggleable
components remain available from FACE3:

- DCT image-frequency gains;
- FFT phase;
- polar radial/twist warp;
- B-spline/Bezier-style free-form deformation;
- barrel and pincushion lens distortion;
- Mobius warp;
- Laplacian-diffused control-grid warp;
- geodesic-inspired 2D face deformation;
- differential-surface-gradient warp.

Edit [`configs/geometry_default.json`](configs/geometry_default.json) to enable
components and change limits. A deterministic `small_random` initialization is
the default because exact identity at neutral geometry gives a stationary
cosine-similarity objective.

[`configs/geometry_extended_all.json`](configs/geometry_extended_all.json)
enables all extended spatial components together for a dedicated sanity run.
Pass `--init small_random` when using that preset so the cosine objective does
not begin at its exact-identity stationary point.

## Cases

The four cases are unchanged:

- `face_002 + add black sunglasses`
- `face_002 + add headphones`
- `face_005 + add black sunglasses`
- `face_005 + add headphones`

Inputs are resolved from MAT, preferring `data/<face_id>/instruct_512.png`.

## A6000 setup and correctness check

```bash
cd /home/interns/Desktop/face4
git pull origin main

bash scripts/download_arcface_checkpoint_a6000.sh /home/interns/Desktop/face4

$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face4.scripts.check_instruct_parity \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face4/models/arcface/iresnet100.pth \
  --edit-steps 2
```

Do not start a long run unless that command exits successfully with every
parity check set to `true`.

## Quick smoke

```bash
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face4.scripts.smoke_timing \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face4/models/arcface/iresnet100.pth \
  --iters 2 \
  --edit-steps 2 \
  --quick \
  --geometry-config configs/geometry_default.json
```

Then run all four smoke cases with `--all-cases`.

The effective loss scale and any same-state retries are recorded in every
history row. A retry does not advance Adam or alter the geometry parameters.

## Full matrix

Use a real edit length (normally 20 steps) after the smoke passes:

```bash
mkdir -p logs
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face4.scripts.run_matrix \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face4/models/arcface/iresnet100.pth \
  --geometry-config configs/geometry_default.json \
  --output-root outputs/edited_output_identity_exact \
  --iters 1800 \
  --edit-steps 20 \
  --lr 0.1 \
  --skip-deepface \
  2>&1 | tee logs/face4_exact_1800.log
```

## Output semantics

`history.csv` and `history.jsonl` contain the exact stock-equivalent Z used for
backpropagation. `Z_stock_validation` is populated at validation states and
must agree within tolerance. `summary.json` keeps these quantities separate:

- `best_Z` / `final_Z`: exact gradient-capable pipeline forward;
- `best_Z_stock_public` / `final_Z_stock_public`: normal PIL stock replay;
- `best_Z_stock_gap` / `final_Z_stock_gap`: their absolute differences.

The public `original_edited.png`, `perturbed_best_edited.png`, and
`perturbed_final_edited.png` always come from the ordinary decorated stock
pipeline. Exact-path counterparts are also saved for direct parity inspection.

Parity uses two separate tolerances. Grad-enabled, no-grad, and decorated
stock **tensor** forwards consume the same canonical tensor and therefore use
the strict `parity_exact_max_Z_gap` tolerance (default `1e-6`). Ordinary PIL
public replay includes image save/reload and Diffusers PIL postprocessing; it
retains the image-SSIM gate and uses `parity_native_pil_max_Z_gap` (default
`0.005`). The deprecated `--parity-max-z-gap` option, if supplied, overrides
only the PIL/public-replay tolerance and never weakens exact tensor parity.

`config_resolved.json` is written before parity preflight begins, so failed
runs preserve the exact enabled components, limits, initialization, and
thresholds that produced the failure.

Model checkpoints and local resume tensors are ignored by Git.

## Identity-metric limitation

The current experiment retains FACE3's full-image resize before ArcFace so the
case definition remains comparable. Standard ArcFace deployments normally use
detected and aligned 112x112 face crops. FACE4 fixes checkpoint/model
correctness and stock-forward parity, but the full-image preprocessing choice
should still be stated when interpreting absolute cosine values.


$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face4.scripts.summarize_runs \
  --results-root outputs/edited_output_identity_exact \
  --output-root outputs/reports/edited_output_identity_exact