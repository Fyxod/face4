# FACE4 smoke_timing summary

- status: failed
- experiment: edited_output_identity
- execution: sequential
- iterations per run: 2
- InstructPix2Pix edit steps in gradient loop: 20
- runs attempted: 1
- runs completed: 0
- failures: 1
- wall seconds: 32.90
- timing estimates: unavailable (no completed runs; timing estimates are unavailable)
- peak VRAM GB: 15.418688297271729
- all required per-iteration fields populated: True
- clamp/project logic active: True

## Failures

- edited_output_identity__face_002__add_black_sunglasses: CorrectnessGateError("Editor parity preflight failed: {'finite_nonzero_input_gradient': True, 'grad_no_grad_exact_forward_parity': True, 'grad_stock_tensor_exact_forward_parity': True, 'stock_tensor_native_pil_parity': True, 'Z_parity': False}")