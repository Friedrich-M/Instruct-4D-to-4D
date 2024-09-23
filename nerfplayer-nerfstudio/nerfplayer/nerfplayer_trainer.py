# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to train model, only needed in order to not save InstructPix2Pix checkpoints
"""
from dataclasses import dataclass, field
from typing import Type, Dict, Tuple, cast
import functools
import torch
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.viewer.server.viewer_elements import ViewerButton
from nerfstudio.utils.decorators import check_main_thread

import time
from tqdm import tqdm
from dataclasses import dataclass, field
from rich import box, style
from rich.panel import Panel
from rich.table import Table

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.callbacks import TrainingCallbackLocation
from nerfstudio.utils.decorators import check_main_thread
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE

import threading
import datetime, os, sys
from PIL import Image
import imageio
import shutil

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]

@dataclass
class NerfplayerTrainerConfig(TrainerConfig):
    """Configuration for the InstructNeRF2NeRFTrainer."""
    _target: Type = field(default_factory=lambda: NerfplayerTrainer)
    """target class to instantiate"""
    render_mode: bool = False
    """Whether to render the scene or not"""
    
class NerfplayerTrainer(Trainer):
    """Trainer for InstructNeRF2NeRF"""

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:

        super().__init__(config, local_rank, world_size)

        # reset button
        self.reset_button = ViewerButton(name="Reset Button", cb_hook=self.reset_callback)

    def reset_callback(self, handle: ViewerButton) -> None:
        """Reset the model to the original checkpoint"""
        
        # load checkpoint
        self._load_checkpoint()

        # reset dataset
        self.config.pipeline.datamanager.image_batch['image'] = self.config.pipeline.datamanager.original_image_batch['image'].clone()
        self.config.pipeline.datamanager.image_batch['image_idx'] = self.config.pipeline.datamanager.original_image_batch['image_idx'].clone()
        
    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        # don't want to call save_dataparser_transform if pipeline's datamanager does not have a dataparser
        if isinstance(self.pipeline.datamanager, VanillaDataManager):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )
            
        if self.config.render_mode:
            self.render_video()
            shutil.rmtree(self.base_dir) # delete checkpoint dir
            sys.exit()
            
        self._init_viewer_state()
        
        num_iterations = self.config.max_num_iterations
        step = 0
        self.pipeline.init_start_step(self._start_step) 
        self.pipeline.init_end_step(self._start_step + num_iterations) 
        
        ########## Debugging ##########
        # self.pipeline.test_edit()
        # import sys; sys.exit()
        ##########    EXIT   ##########
        
        self.dataset_update_thread = threading.Thread(target=self.pipeline.test_edit, name="Dataset Update Thread") # dataset update thread
        self.dataset_update_thread.start() # start dataset update thread
        
        pbar = tqdm(range(self._start_step, self._start_step + num_iterations), desc="Training")
        for step in pbar:
            while self.training_state == "paused":
                time.sleep(0.01)
            with self.train_lock:
                self.pipeline.train()

                # training callbacks before the training iteration
                for callback in self.callbacks:
                    callback.run_callback_at_location(
                        step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                    )

                # time the forward pass
                loss, loss_dict, metrics_dict = self.train_iteration(step)

                # training callbacks after the training iteration
                for callback in self.callbacks:
                    callback.run_callback_at_location(
                        step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                    )
                    
                pbar.set_postfix(loss=loss.item())

            self._update_viewer_state(step)

            # Do not perform evaluation if there are no validation images
            if self.pipeline.datamanager.eval_dataset:
                self.eval_iteration(step)

            if step_check(step, self.config.steps_per_save):
                self.save_checkpoint(step)

        self.dataset_update_thread.join()

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # render video at the end of training
        self.render_video(sample_frame_rate=1)

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers
        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        pipeline_state_dict = {k: v for k, v in self.pipeline.state_dict().items() if "ip2p." not in k}
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else pipeline_state_dict,
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()
                    
    def render_video(self, n_sample_frames: int = None, sample_frame_rate: int = 1):
        frames_dir = os.path.join('./video_frames', self.config.experiment_name)
        os.makedirs(frames_dir, exist_ok=True)
        video_dir = os.path.join('./render_videos', self.config.experiment_name)
        os.makedirs(video_dir, exist_ok=True)
        
        # set model to eval mode
        self.pipeline.model.training = False
        self.pipeline.model.eval()
        
        tag = self.pipeline.config.prompt.split(' ')[-1].replace('?','') 
        
        frames = []
        cameras = self.pipeline.datamanager.train_dataparser_outputs.cameras
        sample_index = list(range(0, cameras.shape[0], sample_frame_rate))
        sample_index = sample_index[:n_sample_frames] if n_sample_frames else sample_index
        
        for img_idx in tqdm(sample_index, desc=f"Rendering scene"):
            img_idx = torch.tensor(img_idx)
            camera_transforms = self.pipeline.datamanager.train_camera_optimizer(img_idx.unsqueeze(dim=0))
            current_camera = cameras[img_idx].to(self.device)
            current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)
            camera_outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
            
            preds_rgb = (
                camera_outputs['rgb']
                .cpu()
                .clamp(0, 1)
                .mul(255.0)
                .byte()
                .numpy()
            )
            
            frame_dir_path = os.path.join(frames_dir, tag)
            os.makedirs(frame_dir_path, exist_ok=True)
            Image.fromarray(preds_rgb).save(f"{frame_dir_path}/{img_idx.item()}.png")
            frames.append(preds_rgb)

        video_dir_path = os.path.join(video_dir, tag)
        os.makedirs(video_dir_path, exist_ok=True)
        video_path = os.path.join(video_dir_path, f"{str(datetime.datetime.now().strftime('%d_%H%M'))}_rendering.mp4")
        imageio.mimwrite(video_path, frames, fps=30)
        
        CONSOLE.print(f"Saved rendering path with {len(frames)} frames to {video_path}")
                
                
