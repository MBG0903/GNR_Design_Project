# Vendored ControlNet training example

Source: https://github.com/huggingface/diffusers/blob/v0.35.1/examples/controlnet/train_controlnet.py

Upstream SHA-256: `a8e6fb260c4efed58027ffafba72abaed73e31f0e0ad80dbf753301a6e4d2481`

Licensed under Apache-2.0; see LICENSE and the preserved source header.

Local changes: load the prepared Hugging Face Dataset from disk; replace bilinear resizing/centre cropping with an exact-resolution assertion. All optimizer, diffusion-loss, checkpointing and model-save logic remains upstream. Training uses frozen Stable Diffusion backbone/VAE/text encoder and updates the ControlNet parameters.
