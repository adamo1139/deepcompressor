# -*- coding: utf-8 -*-
"""Collect calibration dataset."""

import os
from dataclasses import dataclass

import datasets
import torch
from omniconfig import configclass
from torch import nn
from tqdm import tqdm
from PIL import Image
import numpy as np

from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.utils.common import hash_str_to_int, tree_map

from ...utils import get_control
from ..data import get_dataset
from .utils import CollectHook


def process(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    return torch.from_numpy(x.float().numpy()).to(dtype)


def create_dummy_image(size=(1104, 832), mode='black'):
    """Create a simple dummy image for Qwen Image Edit calibration.
    
    Args:
        size (tuple): Image size as (width, height)
        mode (str): Image mode ('black', 'white', 'noise')
    
    Returns:
        PIL.Image: Dummy image in RGB format
    """
    if mode == 'black':
        return Image.new('RGB', size, color='black')
    elif mode == 'white':
        return Image.new('RGB', size, color='white')
    elif mode == 'noise':
        # Create random noise image
        array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        return Image.fromarray(array)
    else:
        return Image.new('RGB', size, color='black')


def collect(config: DiffusionPtqRunConfig, dataset: datasets.Dataset):
    samples_dirpath = os.path.join(config.output.root, "samples")
    caches_dirpath = os.path.join(config.output.root, "caches")
    os.makedirs(samples_dirpath, exist_ok=True)
    os.makedirs(caches_dirpath, exist_ok=True)
    caches = []

    # Build pipeline with for_calibration=True for Qwen Image Edit
    if config.pipeline.name == "qwen-image-edit":
        pipeline = config.pipeline.build(for_calibration=True)
    else:
        pipeline = config.pipeline.build()
    model = pipeline.unet if hasattr(pipeline, "unet") else pipeline.transformer
    assert isinstance(model, nn.Module)
    model.register_forward_hook(CollectHook(caches=caches), with_kwargs=True)

    batch_size = config.eval.batch_size
    print(f"In total {len(dataset)} samples")
    print(f"Evaluating with batch size {batch_size}")
    pipeline.set_progress_bar_config(desc="Sampling", leave=False, dynamic_ncols=True, position=1)
    for batch in tqdm(
        dataset.iter(batch_size=batch_size, drop_last_batch=False),
        desc="Data",
        leave=False,
        dynamic_ncols=True,
        total=(len(dataset) + batch_size - 1) // batch_size,
    ):
        filenames = batch["filename"]
        prompts = batch["prompt"]
        seeds = [hash_str_to_int(name) for name in filenames]
        generators = [torch.Generator(device=pipeline.device).manual_seed(seed) for seed in seeds]
        pipeline_kwargs = config.eval.get_pipeline_kwargs()

        task = config.pipeline.task
        control_root = config.eval.control_root
        
        # Handle Qwen Image Edit as text-to-image during calibration
        if config.pipeline.name == "qwen-image-edit":
            # For Qwen Image Edit during calibration, create dummy images
            # since the pipeline requires an image input even during calibration
            dummy_images = [create_dummy_image() for _ in range(len(prompts))]
            pipeline_kwargs["image"] = dummy_images
        elif task in ["canny-to-image", "depth-to-image", "inpainting"]:
            controls = get_control(
                task,
                batch["image"],
                names=batch["filename"],
                data_root=os.path.join(
                    control_root, config.collect.dataset_name, f"{dataset.config_name}-{config.eval.num_samples}"
                ),
            )
            if task == "inpainting":
                pipeline_kwargs["image"] = controls[0]
                pipeline_kwargs["mask_image"] = controls[1]
            else:
                pipeline_kwargs["control_image"] = controls

        result_images = pipeline(prompts, generator=generators, **pipeline_kwargs).images
        num_guidances = (len(caches) // batch_size) // config.eval.num_steps
        num_steps = len(caches) // (batch_size * num_guidances)
        assert (
            len(caches) == batch_size * num_steps * num_guidances
        ), f"Unexpected number of caches: {len(caches)} != {batch_size} * {config.eval.num_steps} * {num_guidances}"
        for j, (filename, image) in enumerate(zip(filenames, result_images, strict=True)):
            image.save(os.path.join(samples_dirpath, f"{filename}.png"))
            for s in range(num_steps):
                for g in range(num_guidances):
                    c = caches[s * batch_size * num_guidances + g * batch_size + j]
                    c["filename"] = filename
                    c["step"] = s
                    c["guidance"] = g
                    c = tree_map(lambda x: process(x), c)
                    torch.save(c, os.path.join(caches_dirpath, f"{filename}-{s:05d}-{g}.pt"))
        caches.clear()


@configclass
@dataclass
class CollectConfig:
    """Configuration for collecting calibration dataset.

    Args:
        root (`str`, *optional*, defaults to `"datasets"`):
            Root directory to save the collected dataset.
        dataset_name (`str`, *optional*, defaults to `"qdiff"`):
            Name of the collected dataset.
        prompt_path (`str`, *optional*, defaults to `"prompts/qdiff.yaml"`):
            Path to the prompt file.
        num_samples (`int`, *optional*, defaults to `128`):
            Number of samples to collect.
    """

    root: str = "datasets"
    dataset_name: str = "qdiff"
    data_path: str = "prompts/qdiff.yaml"
    num_samples: int = 128


if __name__ == "__main__":
    parser = DiffusionPtqRunConfig.get_parser()
    parser.add_config(CollectConfig, scope="collect", prefix="collect")
    configs, _, unused_cfgs, unused_args, unknown_args = parser.parse_known_args()
    ptq_config, collect_config = configs[""], configs["collect"]
    assert isinstance(ptq_config, DiffusionPtqRunConfig)
    assert isinstance(collect_config, CollectConfig)
    if len(unused_cfgs) > 0:
        print(f"Warning: unused configurations {unused_cfgs}")
    if unused_args is not None:
        print(f"Warning: unused arguments {unused_args}")
    assert len(unknown_args) == 0, f"Unknown arguments: {unknown_args}"

    collect_dirpath = os.path.join(
        collect_config.root,
        str(ptq_config.pipeline.dtype),
        ptq_config.pipeline.name,
        ptq_config.eval.protocol,
        collect_config.dataset_name,
        f"s{collect_config.num_samples}",
    )
    print(f"Saving caches to {collect_dirpath}")

    dataset = get_dataset(
        collect_config.data_path,
        max_dataset_size=collect_config.num_samples,
        return_gt=ptq_config.pipeline.task in ["canny-to-image"],
        repeat=1,
    )

    ptq_config.output.root = collect_dirpath
    os.makedirs(ptq_config.output.root, exist_ok=True)
    collect(ptq_config, dataset=dataset)
