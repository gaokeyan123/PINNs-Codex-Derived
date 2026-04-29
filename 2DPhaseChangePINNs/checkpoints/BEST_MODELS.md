# Best Model Checkpoints

This file is tracked by git. The `.pth` weight files are **not** tracked (see root `.gitignore`).

## Naming convention

```
pinns_v{MAJOR}.{MINOR}_ep{EPOCH:05d}_loss{VAL:.4e}.pth
```

| Filename | Val Loss | Epoch | Notes |
|---|---|---|---|
| `pinns_v1.0_ep00002_loss2.1753e+02.pth` | 2.175283e+02 | 2 | Best phase1 checkpoint from training run |
