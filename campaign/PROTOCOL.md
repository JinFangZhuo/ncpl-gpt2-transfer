# Finite 20-configuration GPT-2 data campaign

- Frozen: 2026-09-02T11:52:42.021370-04:00
- New configurations: 20
- Optimizer updates per run: 5120
- Training seed: 0
- GPUs per run: 2-GPU DDP on physical GPUs 0 and 3
- Gradient accumulation: 64, preserving global batch 128 per configuration
- Historical exact configurations excluded: 197
- Historical optimizer signatures excluded: 171
- Duplicate rule: no full-config duplicate and no duplicate over the eight AdamW/schedule fields
- Design: outcome-blind 20-point maximin Latin hypercube inside the released NCPL domain
- Lifecycle: finite and restart-safe; stop after 20 valid 5,120-update results
