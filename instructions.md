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

The full command is documented in `README.md`; do not run it until the
correctness and timing smokes have been inspected.
