import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List
import sys
pwd=os.getcwd()
sys.path.append(pwd)
import av
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid
from tools import align
from tools.utils  import read_pts_from_jsonfile_compatible
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=784)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    assert os.path.isfile(config.motion_module_path), f"{config.motion_module_path} not exsist"
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
        mm_zero_proj_out=False,  #True - test stage1
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    for ref_image_path in config["test_cases"].keys():
        # Each ref_image may correspond to multiple actions
        for pose_video_path in config["test_cases"][ref_image_path]:
            pose_subdir = None 
            if pose_subdir is not None:
                pose_video_path = pose_video_path.replace('/dance_pose/', pose_subdir)

            ref_name = Path(ref_image_path).stem
            pose_name = Path(pose_video_path).stem.replace("_kps", "")
            
            ref_image_pil = Image.open(ref_image_path).convert("RGB")

            pose_list = []
            pose_tensor_list = []
            pose_images = read_frames(pose_video_path)

            src_fps = get_fps(pose_video_path)

            print(f"pose video has {len(pose_images)} frames, with {src_fps} fps, {ref_image_path}, {pose_video_path}")

            # read pts info and align
            ref_image_pts = read_pts_from_jsonfile_compatible(ref_image_path+'.json')["info_dict_list"]
            pose_video_pts =  read_pts_from_jsonfile_compatible(pose_video_path+'.json')["info_dict_list"]

            max_fps = 30
            if max_fps < src_fps:
                print(f"src_fps={src_fps}, max_fps={max_fps}")
                ratio = max_fps/src_fps
                
                pose_images_new = []
                pose_video_pts_new = []
                new_frame_num = int(len(pose_images)*ratio)
                for i in range(new_frame_num):
                    pose_images_new.append(pose_images[int(i/ratio)])
                    pose_video_pts_new.append(pose_video_pts[int(i/ratio)])

                src_fps = max_fps
                pose_images = pose_images_new
                pose_video_pts = pose_video_pts_new

            # add align
            ref_image_pil, pose_images = align.process(ref_image_pil, ref_image_pts, pose_images, pose_video_pts, dst_size=(width, height))

            pose_transform = transforms.Compose(
                [transforms.Resize((height, width)), transforms.ToTensor()]
            )
            for pose_image_pil in pose_images[: args.L]:
                pose_tensor_list.append(pose_transform(pose_image_pil))
                pose_list.append(pose_image_pil)


            ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
            ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
            ref_image_tensor = repeat(
                ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=args.L
            )

            pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
            pose_tensor = pose_tensor.transpose(0, 1)
            pose_tensor = pose_tensor.unsqueeze(0)

            video = pipe(
                ref_image_pil,
                pose_list,
                width,
                height,
                args.L,
                args.steps,
                args.cfg,
                generator=generator,
            ).videos
            

            video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
            n_rows = 3

            save_videos_grid(
                video,
                f"{save_dir}/{ref_name}_{pose_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4",
                n_rows=n_rows,
                fps=args.fps,
            )


if __name__ == "__main__":
    main()

