from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig


from my_method.data.my_method_datamanager import MyMethodDataManagerConfig
from my_method.my_method import MyMethodModelConfig
from my_method.my_method_pipeline import MyMethodPipelineConfig

MyMethod = MethodSpecification(config=TrainerConfig(
    method_name="my-method",
    pipeline=MyMethodPipelineConfig(
            datamanager=MyMethodDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=MyMethodModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(12, 12),
                hashgrid_resolutions=((16, 128), (128, 512)),
                num_lerf_samples=24,
            ),
    optimizers={
        "test_optimizer": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,    
        }
    }
    ,
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
  ),
  description="My own method testing"
  )
)