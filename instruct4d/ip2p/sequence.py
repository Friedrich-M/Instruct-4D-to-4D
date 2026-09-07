"""Anchor-aware InstructPix2Pix.

Vanilla InstructPix2Pix edits one image at a time, so editing a batch of frames
independently produces a different stylisation for each frame.  Section 3.2 of
the paper restores *within-batch consistency* by inflating the 2D UNet into a
pseudo-3D one (:mod:`instruct4d.ip2p.unet`) and replacing self-attention with
anchor-aware attention (:mod:`instruct4d.ip2p.attention`), so that every frame
in the batch attends to a shared anchor frame.

The module weights are still the stock ``timbrooks/instruct-pix2pix``
checkpoint: only the attention/convolution wiring changes, so no finetuning is
required.

Typical use::

    ip2p = SequenceInstructPix2Pix(device="cuda:1")
    edited = ip2p.edit_sequence(frames, originals, prompt="Make it snow")

where ``frames`` are the current NeRF renders and ``originals`` are the
corresponding unedited training images used as the image condition.
"""

from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from rich.console import Console
from torch import Tensor, nn

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from .attention import DEFAULT_SELF_ATTENTION, SELF_ATTENTION_TYPES
from .inversion import ddim_inversion
from .pipeline import InstructPix2PixPipeline
from .unet import UNet3DConditionModel

CONSOLE = Console(width=120)

#: Scale factor between the VAE latent space and image space (Stable Diffusion v1).
LATENT_SCALE = 0.18215

#: Weights are pulled from these Hugging Face repositories on first use.
IP2P_SOURCE = "timbrooks/instruct-pix2pix"
DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"

#: How the latents entering the denoising loop are initialised.
#:
#: ``noisy_latent``  Noise the encoded input frames up to ``T``.  This keeps the
#:                   current NeRF renders as the starting point and is what both
#:                   editing pipelines use.
#: ``noise``         Start from pure Gaussian noise, ignoring the input frames.
#: ``ddim_inv``      Recover the latents by DDIM inversion of the input frames.
LATENT_INIT_MODES = ("noisy_latent", "noise", "ddim_inv")


class SequenceInstructPix2Pix(nn.Module):
    """Edit a whole batch of frames in one diffusion pass, consistently.

    Args:
        device: Device holding the diffusion model.  Editing and NeRF
            optimisation run concurrently, so this is normally a *different*
            GPU from the one training the radiance field.
        use_full_precision: Run the UNet and VAE in fp32 instead of fp16.
            Slower and roughly twice the memory, but avoids fp16 artefacts.
        self_attention: Which anchor-aware scheme the UNet uses; a key of
            :data:`~instruct4d.ip2p.attention.SELF_ATTENTION_TYPES`.  The
            multi-view pipeline uses the default ``"anchor"``, which ties every
            frame to the anchor alone.  The single-view pipeline uses
            ``"anchor_self"``, letting each frame keep more of its own content
            because consecutive frames differ more there.
    """

    def __init__(
        self,
        device: Union[torch.device, str],
        use_full_precision: bool = False,
        self_attention: str = DEFAULT_SELF_ATTENTION,
    ) -> None:
        super().__init__()

        if self_attention not in SELF_ATTENTION_TYPES:
            raise ValueError(
                f"self_attention must be one of {sorted(SELF_ATTENTION_TYPES)}, got {self_attention!r}"
            )

        CONSOLE.print(f"Loading anchor-aware InstructPix2Pix (self_attention={self_attention})...")
        self.device = device
        self.weights_dtype = torch.float32 if use_full_precision else torch.float16

        tokenizer = CLIPTokenizer.from_pretrained(IP2P_SOURCE, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(IP2P_SOURCE, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(IP2P_SOURCE, subfolder="vae")
        # Inflates the pretrained 2D UNet weights into the pseudo-3D UNet.
        unet = UNet3DConditionModel.from_pretrained_2d(
            IP2P_SOURCE, subfolder="unet", self_attention=self_attention
        )

        for module in (vae, text_encoder, unet):
            module.requires_grad_(False)
            module.to(self.device, dtype=self.weights_dtype)

        self.pipe = InstructPix2PixPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler"),
        )
        self.vae = vae
        self.unet = unet
        self.scheduler = self.pipe.scheduler
        # A second scheduler instance so DDIM inversion cannot disturb the
        # timestep state of the forward denoising loop.
        self.ddim_inv_scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")

        CONSOLE.print("Anchor-aware InstructPix2Pix loaded!")

    @torch.no_grad()
    def edit_sequence(
        self,
        images: Float[Tensor, "F 3 H W"],
        images_cond: Float[Tensor, "F 3 H W"],
        prompt: str = "",
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        noisy_latent_type: str = "noisy_latent",
        T: int = 1000,
    ) -> Float[Tensor, "F 3 H W"]:
        """Edit ``images`` as one consistent batch.

        Frame 0 acts as the anchor: the attention layers let every other frame
        read its keys and values, which is what keeps the stylisation coherent
        across the batch.  Callers are therefore expected to place the frame
        they want to propagate from at index 0.

        Args:
            images: Frames to edit, in ``[0, 1]``.  These are the current NeRF
                renders, i.e. the starting point of the diffusion trajectory.
            images_cond: Matching *unedited* training images, in ``[0, 1]``.
                InstructPix2Pix conditions on these so the edit stays anchored
                to the real scene content.
            prompt: The editing instruction.
            guidance_scale: Text classifier-free guidance weight.
            image_guidance_scale: Image classifier-free guidance weight.  Larger
                values keep the result closer to ``images_cond``.
            diffusion_steps: Number of denoising steps.
            noisy_latent_type: One of :data:`LATENT_INIT_MODES`.
            T: Timestep to noise up to.  Lower values preserve more of
                ``images``, which is how the refinement passes stay faithful to
                an already-warped frame.

        Returns:
            The edited frames in ``[0, 1]``, same shape as ``images``.
        """
        if noisy_latent_type not in LATENT_INIT_MODES:
            raise ValueError(
                f"noisy_latent_type must be one of {LATENT_INIT_MODES}, got {noisy_latent_type!r}"
            )

        sequence_length, _, height, width = images.shape
        # The VAE downsamples by 8, so both sides must be a multiple of 8.
        latent_height, latent_width = height // 8 * 8, width // 8 * 8

        images = self._to_model_input(images, latent_height, latent_width)
        images_cond = self._to_model_input(images_cond, latent_height, latent_width)

        latents = self.imgs_to_latent(images)
        image_latents = self.prepare_image_latents(images_cond)

        # The pseudo-3D UNet expects an explicit frame axis: (B, C, F, h, w).
        latents = rearrange(latents, "f c h w -> 1 c f h w")
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        images_cond = rearrange(images_cond, "f c h w -> 1 f c h w")

        self.set_num_train_timesteps(T)
        self.scheduler.set_timesteps(diffusion_steps)
        self.ddim_inv_scheduler.set_timesteps(diffusion_steps)

        if noisy_latent_type == "noise":
            # `None` makes the pipeline sample fresh Gaussian latents itself.
            latents = None
        elif noisy_latent_type == "noisy_latent":
            latents = self.scheduler.add_noise(
                latents, torch.randn_like(latents), self.scheduler.timesteps[0]
            )
        else:  # ddim_inv
            latents = ddim_inversion(
                self.pipe,
                self.ddim_inv_scheduler,
                latent=latents,
                image_latents=image_latents,
                num_inv_steps=diffusion_steps,
                prompt=prompt,
            )[-1].to(self.device, dtype=self.weights_dtype)

        edited = self.pipe(
            prompt=prompt,
            video_length=sequence_length,
            image=images_cond,
            latents=latents,
            height=latent_height,
            width=latent_width,
            num_inference_steps=diffusion_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
        ).videos  # (1, 3, F, h, w)

        edited = rearrange(edited, "1 c f h w -> f c h w")
        if edited.shape[-2:] != (height, width):
            edited = F.interpolate(edited, size=(height, width), mode="bilinear", align_corners=False)
        return edited

    def _to_model_input(
        self, images: Float[Tensor, "F 3 H W"], height: int, width: int
    ) -> Float[Tensor, "F 3 h w"]:
        """Resize to a VAE-compatible resolution and move onto the IP2P device."""
        if images.shape[-2:] != (height, width):
            images = F.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)
        return images.to(self.device, dtype=self.weights_dtype)

    def imgs_to_latent(self, imgs: Float[Tensor, "F 3 H W"]) -> Float[Tensor, "F 4 h w"]:
        """Encode ``[0, 1]`` images into sampled VAE latents."""
        posterior = self.vae.encode(2 * imgs - 1).latent_dist
        return posterior.sample() * LATENT_SCALE

    def prepare_image_latents(self, imgs: Float[Tensor, "F 3 H W"]) -> Float[Tensor, "F 4 h w"]:
        """Encode the conditioning images, using the distribution mode.

        The condition must be deterministic, hence ``mode()`` rather than
        ``sample()``, and it is *not* rescaled by :data:`LATENT_SCALE` -- the
        UNet consumes it as a raw channel-wise concatenation.
        """
        return self.vae.encode(2 * imgs - 1).latent_dist.mode()

    def latents_to_img(self, latents: Float[Tensor, "F 4 h w"]) -> Float[Tensor, "F 3 H W"]:
        """Decode VAE latents back into ``[0, 1]`` images."""
        imgs = self.vae.decode(latents / LATENT_SCALE).sample
        return (imgs / 2 + 0.5).clamp(0, 1)

    def set_num_train_timesteps(self, num_steps: int = 1000) -> None:
        """Rescale the noise schedule so the trajectory starts at ``num_steps``.

        Shortening the schedule is how the refinement passes apply a light-touch
        edit: less noise is injected, so more of the input frame survives.
        """
        self.scheduler.config.num_train_timesteps = num_steps
        self.ddim_inv_scheduler.config.num_train_timesteps = num_steps

    def forward(self):
        raise NotImplementedError("Use edit_sequence(); this module is not called directly.")
