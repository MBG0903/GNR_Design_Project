# Paired generative fine-tuning: 80:20 experiment

[Open the training Colab notebook](https://colab.research.google.com/github/MBG0903/GNR_Design_Project/blob/main/notebooks/Paired_ControlNet_Training.ipynb).

The recommended first trainable baseline is **Stable Diffusion 1.5 with the pretrained Lineart ControlNet**. ControlNet learns the mapping from an artist's sketch to the matching designer raster. Training on Step_3 alone would teach appearance without establishing which fills correspond to a new sketch. This experiment uses both members of each pair.

Models and upstream implementation:

- [Stable Diffusion 1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Pretrained Lineart ControlNet model card](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart)
- [ControlNet 1.1 lineart documentation](https://github.com/lllyasviel/ControlNet-v1-1-nightly#controlnet-11-lineart), which describes support for manually drawn lineart
- [Pinned Diffusers training example](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/controlnet/train_controlnet.py)

Model cards describe their pretrained-weight licences. The vendored trainer retains Apache-2.0 attribution and licence in `vendor/controlnet/`. It changes only local dataset loading and the image transform: already-prepared native patches must have the exact training size, and are never resized/cropped by the trainer.

## What is trained

All ControlNet parameters are fine-tuned. The Stable Diffusion UNet, VAE and text encoder remain frozen. This updates real model weights; it is not reference prompting, retrieval, or a claim that a chat agent learns permanently from uploaded images. This first experiment does not train a second correction agent or introduce a separate UNet LoRA.

The standard diffusion denoising loss learns visual patterns from the paired targets. There is no explicit semantic staircase-count or gap-width loss. The latent VAE compresses spatial detail, so exact one-pixel geometry can still fail even after training. Palette and output dimensions can be checked deterministically; semantic pieces, two-pixel gaps, front/back anchors and staircase ordering still require inspection. Do not promise 70–80% accuracy before measuring held-out results.

Stable Diffusion 1.5 is a practical initial Colab baseline with a lower compute requirement than larger diffusion models. It is not a claim that this older model is the most capable available. The result should establish whether paired training improves on the same pretrained model before committing to a larger architecture/GPU.

## Dataset and split

The notebook reads `MyDrive/GNR Design Data-Set/Stage_2` and `Stage_3`. The earlier audit failure must be diagnosed and repaired: wrong pairing, unequal dimensions, invalid target colours, missing/duplicate group IDs and unreadable images block training. Original images are never silently thresholded, resized or discarded. The notebook prints error categories and saves detailed CSV diagnostics.

`groups.csv` must group all variants/crops from one original design. Filename-based suggestions are a starting point. Exact duplicate inputs or targets and provided family IDs are merged transitively by the audit. Check pairing alignment, because matching dimensions alone are insufficient. Results are marked provisional while `GROUPS_REVIEWED=False`.

The training preparation replaces the audit's preliminary split labels with approximately **80% train / 20% test by family**, rounded to whole families. This avoids related crops across splits. With 1,700 independent pairs it would be about 1,360 train / 340 test; actual pair counts vary with family sizes. Test targets do not enter training patches or checkpoint selection.

The first fixed experiment evaluates the final 2,000-update model. For hyperparameter tuning, reserve validation families within the training 80%; keep the test 20% out of tuning. Do not select the best checkpoint based on test scores.

## Preserve the native pixel grid

Training inputs are aligned native 512×512 sketch/target crops with white padding where needed. Large designs contribute overlapping patches; small designs are padded. There is no scaling, interpolation, rotation or geometric augmentation. Patch metadata records the original coordinates and padding. Large designs contribute more patches, so training is not equally weighted per original design. Patch edges can truncate motifs; this is a local-pattern training baseline with limited full-design context.

Inference generates the whole native canvas in one pass, padded to multiples of 64, then removes that padding. It does not stitch separately generated tiles. Raw model output is retained, then projected to pure red/white. Exact original width and height are enforced. Native canvases above the configured memory budget stop rather than being resized; sufficiently large images need more GPU memory. Whole-image generation at resolutions larger than the training patches may affect quality and needs evaluation.

## Colab execution and persistence

1. Select a GPU runtime. A T4 with 16 GB is the initial target, with batch 1, gradient accumulation 4, mixed fp16, gradient checkpointing and 8-bit Adam. Live memory fit must be measured; a larger GPU may be needed.
2. Mount Drive, check the family map/alignment and resolve audit errors.
3. Freeze the split and prepare training patches. Only training patches are loaded into the Hugging Face Dataset.
4. Download pretrained weights. Exact Hugging Face model revisions are recorded, and reused on reruns. Model access and downloads still need to succeed in the user's Colab runtime. No OpenAI API key is required, and no image-generation API calls are used.
5. Fine-tune for the configured fixed number of optimizer updates. Checkpoints are saved to Drive every 250 updates; the latest two resumable checkpoints plus the final ControlNet are retained. Allow several GB of Drive space for model and optimizer files. Training can take hours; duration and convergence are unmeasured here.
6. Compare pretrained and trained ControlNet on every held-out test design, with identical prompts, seeds and inference settings. Both receive only the sketch as their image condition. Targets are used afterward for scoring.
7. Use the optional future-sketch cell to load the trained adapter and predict Step_3 without a designer target. It can be run after training without executing the lengthy evaluation cell first, though evaluation is necessary to assess whether the model is useful.

The default experiment lives under `MyDrive/GNR Textile Training/Native_ControlNet_v1`. Keep the same JOB to resume. Changing originals, grouping, training settings or model revisions requires a new experiment folder. Learned model hashes are recorded after successful training. Models are not automatically uploaded to Hugging Face or GitHub.

## Outputs and evidence

The notebook saves learned ControlNet weights, split manifests, source/target crop metadata, raw native predictions, exact-palette PNG/BMP, coloured differences, per-design metrics and mean pixel agreement/red IoU/red Dice. Differences use native coordinates without alignment or target resizing. These percentages are similarity measures, not business-rule pass rates. Review complete contours, gaps and intentional white with the designer.

Local checks exercise grouped 80:20 splitting, duplicate-leakage rejection, exclusion of test pairs from training patches, native crop/pad pixel preservation, no-resize output restoration, held-out-only prediction, metric calculation, resume, notebook syntax and vendored trainer contracts. GPU training/model downloads are not executed by these tests. No trained checkpoint or accuracy result is claimed until the user runs the Colab GPU experiment successfully.
