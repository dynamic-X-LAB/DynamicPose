<p align="center">
    <img src="./assets/dancebear.png" width="100"/>
</p>


<div align="center">
<h1>DynamicPose: A robust image-to-video framework for portrait animation driven by pose sequences</h1>
</div>


# Introduction
We introduce DynamicPose, a simple and robust framework for animating human images, specifically designed for portrait animation driven by human pose sequences. In summary, our key contributions are as follows:

1. <u>_**Large-Scale Motion:**_</u> Our model supports large-scale motion in diverse environments and generalizes well to non-realistic scenes, such as cartoons.
2. <u>_**High-Quality Video Generation:**_</u> The model can generate high-quality video dance sequences from a single photo, outperforming most open-source models in the same domain.
3. <u>_**Accurate Pose Alignment:**_</u> We employed a high-accuracy pose detection algorithm and a pose alignment algorithm, which enables us to maintain pose accuracy while preserving the consistency of human body limbs to the greatest extent possible.
4. <u>_**Comprehensive Code Release:**_</u> We will gradually release the code for data filtering, data preprocessing, data augmentation, model training (DeepSpeed Zero2), as well as optimized inference scripts.

We are committed to providing the complete source code for free and regularly updating DynamicPose. By open-sourcing this technology, we aim to drive advancements in the digital human field and promote the widespread adoption of virtual human technology across various industries. If you are interested in any of the modules, please feel free to email us to discuss further. Additionally, if our work can benefit you, we would greatly appreciate it if you could give us a star ⭐!

# News
- [08/28/2024] 🔥 Release DynamicPose project and pretrained models.
- [08/28/2024] 🔥 Release pose server and pose align algorithm.
- In the coming two weeks, we will release Comfyui and Gradio of DynamicPose.


# Release Plans

- [x] Inference codes and pretrained weights
- [x] Release pose server based FaseAPI and pose align algorithm.
- [ ] Comfyui of DynamicPose.
- [ ] Huggingface Gradio demo.
- [ ] Data cleaning and preprocessing pipeline.
- [ ] Training scripts.

# Demos 
<table class="center">



<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/f11ffe9e-9922-4073-9283-80ff0d618d63" muted="false"></video>

    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/25a99873-8b93-4fdc-a25a-f30a3cdcdd59" muted="false"></video>
    </td>
</tr>





<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/f5dfdd26-eaae-4028-97ed-3d6cff33cbb1" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/48404b0f-8cca-4448-9ad6-91cfb8111acf" muted="false"></video>

    </td>
</tr>

<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/4f15465b-6347-4262-82de-659c3bc10ec2" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/26bb42ea-63f1-4541-b8a4-1e25da26e585" muted="false"></video>
    </td>
</tr>




</table>




# Installation

## Build Environtment

We Recommend a python version `>=3.10` and cuda version `=11.7`. Then build environment as follows:

```shell
# [Optional] Create a virtual env
python -m venv .venv
source .venv/bin/activate
# Install with pip:
pip install -r requirements_min.txt
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0" 
```

## Download weights

You can download weights manually as follows.

**Manually downloading**: You can also download weights manually, which has some steps:

1. Download our trained [weights](https://huggingface.co/DynamicXLAB/DynamicPose/tree/main), which include four parts: `denoising_unet.pth`, `reference_unet.pth`, `pose_guider.pth` and `motion_module.pth`.

2. Download pretrained weight of based models and other components: 
    - [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
    - [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
    - [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)

3. Download rtmpose weights (`rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth`, `tmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth`) and the corresponding scripts from mmpose repository. 

Finally, these weights should be orgnized as follows:

```text
./pretrained_weights/
|-- rtmpose
|   |--rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth
|   |-- rtmw-x_8xb320-270e_cocktail14-384x288.py
|   |-- rtmdet_m_640-8xb32_coco-person.py
|   `-- rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
|-- DWPose
|   |-- dw-ll_ucoco_384.onnx
|   `-- yolox_l.onnx
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- denoising_unet.pth
|-- motion_module.pth
|-- pose_guider.pth
|-- reference_unet.pth
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
`-- stable-diffusion-v1-5
    |-- feature_extractor
    |   `-- preprocessor_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   `-- diffusion_pytorch_model.bin
    `-- v1-inference.yaml
```


# Inference

- stage 1 image inference:
```shell
python -m scripts.pose2img --config ./configs/prompts/animation_stage1.yaml -W 512 -H 768
```

- stage 2 video inference:
```shell
python -m scripts.pose2vid --config ./configs/prompts/animation_stage2.yaml -W 512 -H 784 -L 64
```

- You can refer the format of configs to add your own reference images or pose videos. 
First, extract the keypoints from the input reference images and target pose for alignment:

```shell
python data_prepare/video2pose.py path/to/ref/images path/to/save/results image  #image
```

```shell
python data_prepare/video2pose.py path/to/tgt/videos path/to/save/results video #video
```


# Limitation
This work also has some limitations, which are outlined below:

1. When the input image features a profile face, the model is prone to generating distorted faces.

1. When the background is complex, the model struggles to accurately distinguish between the human body region and the background region.

1. When the input image features a person with objects attached to their hands, such as bags or phones, the model has difficulty deciding whether to include these objects in the generated output



# Acknowledgement
1. We thank [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone) for their technical report, and have refer much to [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) and [diffusers](https://github.com/huggingface/diffusers).
1. We thank open-source components like [dwpose](https://github.com/IDEA-Research/DWPose), [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [rtmpose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose), etc.. 


# License
1. `code`: The code of DynamicPose is released under the MIT License.
2. `other models`: Other open-source models used must comply with their license, such as `stable-diffusion-v1-5`, `dwpose`, `rtmpose`, etc..


# Citation
```bibtex
@software{DynamicPose,
  author = {Yanqin Chen, Changhao Qiao, Zou Bin, Dejia Song},
  title = {DynamicPose: A effective image-to-video framework for portrait animation driven by human pose sequences},
  month = {August},
  year = {2024},
  url = {https://github.com/dynamic-X-LAB/DynamicPose}
}
```

