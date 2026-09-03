# Sketch-to-Image Generation
**Novel Approaches Tackling Sketch To Image Generation**

> Mentor: Prof. Koteswar Jerripothula

---

## Overview

This project explores and compares two paradigms for translating sparse hand-drawn sketches into photorealistic images:

1. **CLIP-Guided Conditioned Stable Diffusion** — a training-free pipeline that combines CLIP-based semantic extraction, prompt refinement, and ControlNet-guided diffusion
2. **GAN-Based Translation** — a custom Pix2Pix-style GAN trained on synthetically generated sketch-image pairs, augmented using CycleGAN

The core challenge in sketch-to-image is bridging a sparse, abstract input (a line drawing) with a rich, detailed image output — without losing structural fidelity.

---

## Approaches

### Approach 1: CLIP-Guided Conditioned Stable Diffusion

A three-stage training-free pipeline leveraging pretrained models.

**Step 1 — CLIP Interrogator**
- The sketch is passed through CLIP's visual encoder into a shared embedding space
- Cosine similarity against a bank of candidate text descriptions retrieves the best-matching base prompt
- No manual prompting required — fully automated sketch-to-text

**Step 2 — Prompt Refinement**
- Strips sketch-referencing vocabulary ("drawing", "lines", "pencil")
- Adds style tokens, composition cues (depth of field, lighting, shadows), and quality boosters
- Outputs a vivid, scene-coherent prompt ready for diffusion

**Step 3 — ControlNet + Stable Diffusion**
- ControlNet enforces structural fidelity by injecting the sketch as a conditioning signal into every denoising step
- Three pretrained variants used depending on sketch quality:
  - **Canny Edge** — clean, precise sketches
  - **Scribble** — rough freehand input
  - **Soft-Noise** — abstract or loose sketches
- Stable Diffusion generates photorealistic textures and lighting from the refined prompt
- Zero-convolution layers ensure ControlNet never corrupts the pretrained SD weights

No training required. All components available via HuggingFace.

---

### Approach 2: GAN-Based Translation

**Dataset Strategy**
- Most domains lack paired (sketch, image) datasets — a fundamental GAN bottleneck
- Synthetic pairs generated using:
  - Canny Edge Detection + Thresholding on real images
  - Photoshop-style sketch filters
  - CycleGAN (image→sketch direction) for bootstrapping additional pairs
- Datasets: ~7,300 human face images (Kaggle) + Sketchy Database (GATech)

**CycleGAN**
- Used for unpaired image-to-image translation
- Image→sketch direction trained first (simpler task — images lose color, depth, texture)
- Resulting generator used to produce synthetic sketch-image pairs for augmenting the main training set

**Custom GAN (Pix2Pix-style)**
- Generator: U-Net architecture with Conv2D downsample + Conv2DTranspose upsample blocks, skip connections, tanh output activation
- Discriminator: PatchGAN — takes sketch + generated/real image pair, outputs patch-level real/fake probabilities
- Loss: Adversarial loss + L1 pixel-wise loss
- Training config: Batch=4 · 128×128 px · 10 epochs · Adam lr=2×10⁻⁴ · ~1,800 steps/epoch · 15+ hrs/run

---

## Individual Contributions

**CLIP-Guided Stable Diffusion** — Vrushabh D Undri & Daksh Pratap Singh
- CLIP-based sketch-to-prompt extraction pipeline
- Prompt refinement engineering (composition enhancers, style tokens, sketch reference removal)
- ControlNet integration (Canny Edge, Scribble, Soft-Noise variants)
- Text-to-image diffusion pipeline end-to-end

**GAN Architecture & Data Synthesis** — Sarvesh Bharambe & Harshvardhan Agarwal
- Synthetic dataset generation via Canny Edge Detection and Photoshop filters
- CycleGAN implementation for semi-supervised data augmentation
- Custom Pix2Pix-style GAN design (U-Net generator + PatchGAN discriminator)
- Training optimization under hardware constraints

**Shared** — All members
- Abstract, introduction, and literature review
- Cross-model evaluation and comparative analysis
- Final observations and conclusions

---

## References

1. Adding Conditional Control to Text-to-Image Diffusion Models — ControlNet original paper
2. Chen et al., Deep Generation of Face Images from Sketches, 2020
3. Chen & Hays, SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis, 2018
4. Georgia Institute of Technology, [Sketchy Database](https://sketchy.eye.gatech.edu/), 2016
5. Ghosh et al., Interactive Sketch Fill: Multiclass Sketch-to-Image Translation, 2019
6. Gupta, [Human Faces Dataset](https://www.kaggle.com/datasets/ashwingupta3012/human-faces), Kaggle, 2023
7. Vinker et al., CLIPasso: Semantically-Aware Object Sketching, 2022
8. Wang, Bau & Zhu, Sketch Your Own GAN, 2021
9. Yang et al., S2FGAN: Semantically Aware Interactive Sketch-to-Face Translation, WACV 2022

---

## Running the Streamlit Interface (GAN-Based Demo)

```bash
# Create and activate virtual environment
python -m venv streamlit_env
.\streamlit_env\Scripts\Activate.ps1

# Install dependencies
pip install streamlit tensorflow opencv-python pillow numpy

# Launch the app
streamlit run streamlit_app.py
```
