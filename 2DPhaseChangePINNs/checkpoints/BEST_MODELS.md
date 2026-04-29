# Best Model Checkpoints

This file is tracked by git. The `.pth` weight files are **not** tracked (see root `.gitignore`).

## Naming convention

```
pinns_v{MAJOR}.{MINOR}_ep{EPOCH:05d}_loss{VAL:.4e}.pth
```

| Filename | Val Loss | Epoch | Notes |
|---|---|---|---|
| `pinns_v1.0_ep00030_loss5.4630e+00.pth` | 5.463017e+00 | 30 | Best phase1 checkpoint from training run |
