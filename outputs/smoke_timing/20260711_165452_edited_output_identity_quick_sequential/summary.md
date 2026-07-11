# FACE4 smoke_timing summary

- status: failed
- experiment: edited_output_identity
- execution: sequential
- iterations per run: 2
- InstructPix2Pix edit steps in gradient loop: 2
- runs attempted: 1
- runs completed: 0
- failures: 1
- wall seconds: 27.50
- timing estimates: unavailable (no completed runs; timing estimates are unavailable)
- peak VRAM GB: 7.68686580657959
- all required per-iteration fields populated: True
- clamp/project logic active: True

## Failures

- edited_output_identity__face_002__add_black_sunglasses: CorrectnessGateError('Non-finite geometry gradients after unscale at state 1; parameter indices=[0, 1, 2] backward_scale=65536.0')