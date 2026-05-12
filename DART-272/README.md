# DART-272

Independent DART-style core generation model for 272-dim HumanML3D features.

This project keeps the core generation recipe:

- history-conditioned VAE encoder/decoder
- latent diffusion denoiser
- autoregressive primitive rollout

It does not depend on `DART-main` at runtime.

## Scope

This implementation targets the core generator only.

Included:

- 272-dim HumanML3D dataset loader
- trainable text encoder
- VAE training
- latent denoiser training
- rollout sampling

Not included yet:

- SMPL reconstruction
- scene/contact optimization
- RL control
- Blender export

## Data Layout

Defaults assume:

```text
../humanml3d_272/
  motion_data/*.npy
  texts/*.txt
  split/{train,val,test}.txt
  mean_std/{Mean.npy,Std.npy}
```

## Environment

The code is written to run in the `conda` environment `mcc`.

## Example Commands

Train VAE:

```bash
conda run -n mcc python DART-272/train_mvae.py \
  --data-root humanml3d_272 \
  --save-dir DART-272/outputs/mvae_h2_f8_r4
```

Train latent denoiser:

```bash
conda run -n mcc python DART-272/train_mld.py \
  --data-root humanml3d_272 \
  --save-dir DART-272/outputs/mld_h2_f8_r4 \
  --mvae-ckpt DART-272/outputs/mvae_h2_f8_r4/checkpoint_last.pt
```

Rollout:

```bash
conda run -n mcc python DART-272/rollout.py \
  --checkpoint DART-272/outputs/mld_h2_f8_r4/checkpoint_last.pt \
  --text-prompt "walk forward*8,turn left*2,walk forward*8"
```
