"""
NeRFPlayer config.
"""


from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.dycheck_dataparser import DycheckDataParserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfplayer.nerfplayer_nerfacto import NerfplayerNerfactoModelConfig
from nerfplayer.nerfplayer_ngp import NerfplayerNGPModelConfig
from nerfplayer.nerfplayer_trainer import NerfplayerTrainerConfig
from nerfplayer.nerfplayer_pipeline import NerfplayerPipelineConfig
from nerfplayer.nerfplayer_in2n import IN2NPipelineConfig
from nerfplayer.nerfplayer_datamanager import (
    NerfplayerDataManager,
    NerfplayerDataManagerConfig, 
)

edit_nerfacto = MethodSpecification(
    config=NerfplayerTrainerConfig(
        method_name="edit-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=1000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=NerfplayerPipelineConfig(
            datamanager=NerfplayerDataManagerConfig(
                _target=NerfplayerDataManager[DepthDataset],
                dataparser=DycheckDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", 
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=NerfplayerNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,
                use_vgg_perceptual=False,
            ),
            ip2p_use_full_precision=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Dynamic NeRF edit with nerfacto backbone.",
)

in2n_nerfacto = MethodSpecification(
    config=NerfplayerTrainerConfig(
        method_name="in2n-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=1000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=IN2NPipelineConfig(
            datamanager=NerfplayerDataManagerConfig(
                _target=NerfplayerDataManager[DepthDataset],
                dataparser=DycheckDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", 
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=NerfplayerNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,
                use_vgg_perceptual=False,
            ),
            ip2p_use_full_precision=True,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Dynamic NeRF edit with nerfacto backbone.",
)

nerfplayer_nerfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfplayer-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[DepthDataset],
                dataparser=DycheckDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=NerfplayerNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="NeRFPlayer with nerfacto backbone.",
)

nerfplayer_ngp = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfplayer-ngp",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=DynamicBatchPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[DepthDataset],
                dataparser=DycheckDataParserConfig(),
                train_num_rays_per_batch=8192,
            ),
            model=NerfplayerNGPModelConfig(
                eval_num_rays_per_chunk=8192,
                grid_levels=1,
                alpha_thre=0.0,
                render_step_size=0.001,
                disable_scene_contraction=True,
                near_plane=0.01,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=64000),
        vis="viewer",
    ),
    description="NeRFPlayer with InstantNGP backbone.",
)
