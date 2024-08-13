# Instruct 4D-to-4D: Editing 4D Scenes as Pseudo-3D Scenes Using 2D Diffusion

This is the official implementation of [Instruct 4D-to-4D](https://immortalco.github.io/Instruct-4D-to-4D/).

![Pipeline](./imgs/pipeline.png)

## Installation

### Environmental Setups

```bash
git clone https://github.com/Friedrich-M/Instruct-4D-to-4D.git
cd Instruct-4D-to-4D
conda create -n instruct4d python=3.8
conda activate instruct4d
pip install -r requirements.txt
```

### Data Preparation

**For multi-view 4D scenes.** The dataset provided in [DyNeRF](https://github.com/facebookresearch/Neural_3D_Video) is used. You can download scenes from [DyNeRF Dataset](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0).

**For real dynamic scenes:** The dataset provided in [HyperNeRF](https://github.com/google/hypernerf) is used. You can download scenes from [Hypernerf Dataset](https://github.com/google/hypernerf/releases/tag/v0.1) and organize them as [Nerfies](https://github.com/google/nerfies#datasets). 

**For monocular 4D scenes:** The dataset provided in [DyCheck](https://github.com/KAIR-BAIR/dycheck) is used. You can download scenes from [DyCheck Dataset](https://drive.google.com/drive/folders/1ZYQQh0qkvpoGXFIcK_j4suon1Wt6MXdZ).

## Training

To edit a 4D scene, you must first train a regular 4D NeRF using your data.
```bash
pass
```

Once you have fully trained your scene, the checkpoints will be saved to the outputs directory. To start training for editing the NeRF, run the following command:
```bash
pass
```

## Framework

### (1) Anchor-Aware Instruct-Pix2Pix (IP2P)

To enable InsturctPix2Pix to simultaneously edit multiple frames with batch consistency, we modify the original InstructPix2Pix to be anchor-aware, reffering to some zero-shot video editing works.

### (2) Key Pseudo-View Editing (Temporal Consistency)

![Flow-guided Sliding Window](./imgs/sliding_window.png)

### (3) Pseudo-View Propagation (Spatial Consistency)

According to the principal of Perspective Transformation, we could use rendered depth from 4D NeRF with the camera parameters to warp the edited pseudo-view to the target view, while maintaining spatial consistency.

## Tips

[1] **2D Editing Quality.** If your edit isn't working as you desire, it is likely because InstructPix2Pix struggles with your images and prompt. We recommend taking one of your training views and trying to edit it in 2D first with InstructPix2Pix, referring to the tips on getting a good edit can be found [here](https://github.com/timothybrooks/instruct-pix2pix#tips).

[2] **4D Scene Representation.** Our framework is general, and therefore, any 4D scene representation adopting RGB observations as supervision can be used. We encourage to extend our editing pipeline to 4D Gaussian Splatting to make the editing more efficient.

## Acknowledgement

We would like to thank [Liangchen Song](https://lsongx.github.io/index.html) for providing the codebase of [NeRFPlayer](https://lsongx.github.io/projects/nerfplayer.html) and helpful discussion. We also sincerely thank [Haque, Ayaan](https://www.ayaanzhaque.me/) for kind discussion about 3D scene editing.

## Citation

You can find our paper on [arXiv](https://arxiv.org/abs/2406.09402).

If you find this code or find the paper useful for your research, please consider citing:

```
@inproceedings{mou2024instruct,
  title={Instruct 4D-to-4D: Editing 4D Scenes as Pseudo-3D Scenes Using 2D Diffusion},
  author={Mou, Linzhan and Chen, Jun-Kun and Wang, Yu-Xiong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20176--20185},
  year={2024}
}
```