# FACE4 A6000 instructions

Run these in order. Stop if the ArcFace setup or parity command exits nonzero.

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

$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face4.scripts.smoke_timing \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face4/models/arcface/iresnet100.pth \
  --geometry-config configs/geometry_default.json \
  --iters 2 \
  --edit-steps 2 \
  --quick
```

Inspect first:

```text
outputs/correctness_check/**/parity_report.json
outputs/correctness_check/**/parity_sheet.png
outputs/smoke_timing/**/summary.json
outputs/smoke_timing/**/runs/**/correctness_preflight/parity_report.json
```

Only after all parity checks pass, use a real 20-step edit smoke:

```bash
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face4.scripts.smoke_timing \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face4/models/arcface/iresnet100.pth \
  --geometry-config configs/geometry_default.json \
  --iters 2 \
  --edit-steps 20 \
  --quick
```

The runner starts with fp16 loss scale `65536` and automatically retries the
same optimizer state at lower scales if a backward pass overflows. Confirm the
result has `status: complete`; retry counts and the effective scale are saved
in `history.jsonl` and `summary.json`.

The full command is documented in `README.md`; do not run it until the
correctness and timing smokes have been inspected.

To test every extended spatial component together, use the dedicated preset
instead of temporarily editing `geometry_default.json`:

```bash
$HOME/.local/bin/micromamba run \
  -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m face4.scripts.smoke_timing \
  --mat-root /home/interns/Desktop/mat \
  --arcface-checkpoint /home/interns/Desktop/face4/models/arcface/iresnet100.pth \
  --geometry-config configs/geometry_extended_all.json \
  --init small_random \
  --iters 2 \
  --edit-steps 20 \
  --quick
```
