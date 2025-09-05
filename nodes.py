from typing import Union, Tuple
from pathlib import Path
import shutil
import os
import gc
import logging

import torch
from PIL import Image
import numpy as np
import trimesh as Trimesh
import meshlib.mrmeshpy as mrmeshpy
from cv2 import cvtColor, COLOR_RGB2BGR, COLOR_BGR2RGB  # noqa
import safetensors

from hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.hy3dshape.postprocessors import (
    FaceReducer,
    FloaterRemover,
    DegenerateFaceRemover,
)

# painting
from hy3dpaint.utils.uvwrap_utils import mesh_uv_wrap
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.hy3dshape.models.autoencoders import ShapeVAE
from hy3dshape.hy3dshape.meshlib import postprocessmesh
from hy3dshape.hy3dshape.checkpoint_pickle import checkpoint_pickle

from hy3dpaint.utils.torchvision_fix import apply_fix

apply_fix()
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
diffusions_dir = os.path.join(comfy_path, "models", "diffusers")

# Replace folder_paths import with compatibility shim


class FolderPathsShim:
    def __init__(self):
        self.base_path = script_directory
        self.models_dir = os.path.join(self.base_path, "models")
        self.output_dir = os.path.join(self.base_path, "output")
        self.input_dir = os.path.join(self.base_path, "input")

    def get_filename_list(self, subfolder):
        path = os.path.join(self.models_dir, subfolder)
        if os.path.exists(path):
            return [
                f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
            ]
        return []

    def get_full_path(self, subfolder, filename):
        return os.path.join(self.models_dir, subfolder, filename)

    def get_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    def get_input_directory(self):
        os.makedirs(self.input_dir, exist_ok=True)
        return self.input_dir

    def get_save_image_path(self, filename_prefix, output_dir):
        # Parse filename_prefix for subfolder
        parts = filename_prefix.split("/")
        if len(parts) > 1:
            subfolder = os.path.join(*parts[:-1])
            prefix = parts[-1]
        else:
            subfolder = ""
            prefix = filename_prefix

        full_output_folder = (
            os.path.join(output_dir, subfolder) if subfolder else output_dir
        )
        os.makedirs(full_output_folder, exist_ok=True)

        # Find next available counter
        existing_files = [
            f for f in os.listdir(full_output_folder) if f.startswith(prefix)
        ]
        counter = len(existing_files) + 1

        return (full_output_folder, prefix, counter, subfolder, filename_prefix)


folder_paths = FolderPathsShim()


# Replace comfy.model_management import with compatibility shim
# Fallback implementation when ComfyUI is not available
class ModelManagementShim:
    OOM_EXCEPTION = RuntimeError

    def get_torch_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def unet_offload_device(self):
        return torch.device("cpu")

    def soft_empty_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


mm = ModelManagementShim()


def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
    MMAP_TORCH_FILES = False
    DISABLE_MMAP = False
    ALWAYS_SAFE_LOAD = False
    if device is None:
        device = torch.device("cpu")
    metadata = None
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        try:
            with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                sd = {}
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    if DISABLE_MMAP:  # TODO: Not sure if this is the best way to bypass the mmap issues
                        tensor = tensor.to(device=device, copy=True)
                    sd[k] = tensor
                if return_metadata:
                    metadata = f.metadata()
        except Exception as e:
            if len(e.args) > 0:
                message = e.args[0]
                if "HeaderTooLarge" in message:
                    raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.".format(message, ckpt)) from e
                if "MetadataIncompleteBuffer" in message:
                    raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.".format(message, ckpt)) from e
            raise e
    else:
        torch_args = {}
        if MMAP_TORCH_FILES:
            torch_args["mmap"] = True

        if safe_load or ALWAYS_SAFE_LOAD:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True, **torch_args)
        else:
            logging.warning("WARNING: loading {} unsafely, upgrade your pytorch to 2.4 or newer to load this file safely.".format(ckpt))
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=checkpoint_pickle)
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            if len(pl_sd) == 1:
                key = list(pl_sd.keys())[0]
                sd = pl_sd[key]
                if not isinstance(sd, dict):
                    sd = pl_sd
            else:
                sd = pl_sd
    return (sd, metadata) if return_metadata else sd


script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
diffusions_dir = os.path.join(comfy_path, "models", "diffusers")


def parse_string_to_int_list(number_string):
    """
    Parses a string containing comma-separated numbers into a list of integers.

    Args:
      number_string: A string containing comma-separated numbers (e.g., "20000,10000,5000").

    Returns:
      A list of integers parsed from the input string.
      Returns an empty list if the input string is empty or None.
    """
    if not number_string:
        return []

    try:
        # Split the string by comma and convert each part to an integer
        int_list = [int(num.strip()) for num in number_string.split(",")]
        return int_list
    except ValueError as e:
        print(
            f"Error converting string to integer: {e}. Please ensure all values are valid numbers."
        )
        return []


def hy3dpaintimages_to_tensor(images):
    tensors = []
    for pil_img in images:
        np_img = np.array(pil_img).astype(np.uint8)
        np_img = np_img / 255.0
        tensor_img = torch.from_numpy(np_img).float()
        tensors.append(tensor_img)
    tensors = torch.stack(tensors)
    return tensors


def _convert_texture_format(
    tex: Union[np.ndarray, torch.Tensor, Image.Image],
    texture_size: Tuple[int, int],
    device: str,
    force_set: bool = False,
) -> torch.Tensor:
    """Unified texture format conversion logic."""
    if not force_set:
        if isinstance(tex, np.ndarray):
            tex = Image.fromarray((tex * 255).astype(np.uint8))
        elif isinstance(tex, torch.Tensor):
            tex_np = tex.cpu().numpy()

            # 2. Handle potential batch dimension (B, C, H, W) or (B, H, W, C)
            if tex_np.ndim == 4:
                if tex_np.shape[0] == 1:
                    tex_np = tex_np.squeeze(0)
                else:
                    tex_np = tex_np[0]

            # 3. Handle data type and channel order for PIL
            if tex_np.ndim == 3:
                if (
                    tex_np.shape[0] in [1, 3, 4]
                    and tex_np.shape[0] < tex_np.shape[1]
                    and tex_np.shape[0] < tex_np.shape[2]
                ):
                    tex_np = np.transpose(tex_np, (1, 2, 0))
                elif (
                    tex_np.shape[2] in [1, 3, 4]
                    and tex_np.shape[0] > 4
                    and tex_np.shape[1] > 4
                ):
                    pass
                else:
                    raise ValueError(
                        f"Unsupported 3D tensor shape after squeezing batch and moving to CPU. "
                        f"Expected (C, H, W) or (H, W, C) but got {tex_np.shape}"
                    )

                if tex_np.shape[2] == 1:
                    tex_np = tex_np.squeeze(2)  # Remove the channel dimension

            elif tex_np.ndim == 2:
                pass
            else:
                raise ValueError(
                    f"Unsupported tensor dimension after squeezing batch and moving to CPU: {tex_np.ndim} "
                    f"with shape {tex_np.shape}. Expected 2D or 3D image data."
                )

            tex_np_uint8 = (tex_np * 255).astype(np.uint8)

            tex = Image.fromarray(tex_np_uint8)

        tex = tex.resize(texture_size).convert("RGB")
        tex = np.array(tex) / 255.0
        return torch.from_numpy(tex).to(device).float()
    else:
        if isinstance(tex, np.ndarray):
            tex = torch.from_numpy(tex)
        elif isinstance(tex, Image.Image):
            tex = torch.from_numpy(np.array(tex).astype(np.float32) / 255.0)
        return tex.to(device).float()


def convert_ndarray_to_pil(texture):
    texture_size = len(texture)
    tex = _convert_texture_format(texture, (texture_size, texture_size), "cuda")
    tex = tex.cpu().numpy()
    processed_texture = (tex * 255).astype(np.uint8)
    pil_texture = Image.fromarray(processed_texture)
    return pil_texture


def get_filename_list(folder_name: str):
    files = [f for f in os.listdir(folder_name)]
    return files


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def convert_tensor_images_to_pil(images):
    pil_array = []

    for image in images:
        pil_array.append(tensor2pil(image))

    return pil_array


class MetaData:
    def __init__(self):
        self.camera_config = None
        self.albedos = None
        self.mrs = None
        self.albedos_upscaled = None
        self.mrs_upscaled = None
        self.mesh_file = None


class Hy3DMeshGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {
                        "tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder"
                    },
                ),
                "image": ("IMAGE", {"tooltip": "Image to generate mesh from"}),
                "steps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Number of diffusion steps",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 1,
                        "max": 30,
                        "step": 0.1,
                        "tooltip": "Guidance scale",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "attention_mode": (["sdpa", "sageattn"], {"default": "sdpa"}),
            },
        }

    RETURN_TYPES = ("HY3DLATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3D21Wrapper"

    def loadmodel(self, model, image, steps, guidance_scale, seed, attention_mode):
        offload_device = mm.unet_offload_device()

        seed = seed % (2**32)

        # from .hy3dshape.hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        # from .hy3dshape.hy3dshape.rembg import BackgroundRemover
        # import torchvision.transforms as T

        model_path = folder_paths.get_full_path("diffusion_models", model)

        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            config_path=os.path.join(
                script_directory, "configs", "dit_config_2_1.yaml"
            ),
            ckpt_path=model_path,
            offload_device=offload_device,
            attention_mode=attention_mode,
        )

        # to_pil = T.ToPILImage()
        # image = to_pil(image[0].permute(2, 0, 1))

        # if image.mode == 'RGB':
        # rembg = BackgroundRemover()
        # image = rembg(image)

        image = tensor2pil(image)

        latents = pipeline(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed),
        )

        del pipeline
        # del vae

        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

        return (latents,)


class Hy3DMultiViewsGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "camera_config": ("HY3D21CAMERA",),
                "view_size": (
                    "INT",
                    {"default": 512, "min": 512, "max": 1024, "step": 256},
                ),
                "image": ("IMAGE", {"tooltip": "Image to generate mesh from"}),
                "steps": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Number of steps",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1,
                        "max": 10,
                        "step": 0.1,
                        "tooltip": "Guidance scale",
                    },
                ),
                "texture_size": (
                    "INT",
                    {"default": 1024, "min": 512, "max": 4096, "step": 512},
                ),
                "unwrap_mesh": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = (
        "HY3DPIPELINE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "HY3D21CAMERA",
        "HY3D21METADATA",
    )
    RETURN_NAMES = (
        "pipeline",
        "albedo",
        "mr",
        "positions",
        "normals",
        "camera_config",
        "metadata",
    )
    FUNCTION = "genmultiviews"
    CATEGORY = "Hunyuan3D21Wrapper"

    def genmultiviews(
        self,
        trimesh,
        camera_config,
        view_size,
        image,
        steps,
        guidance_scale,
        texture_size,
        unwrap_mesh,
        seed,
    ):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        seed = seed % (2**32)

        conf = Hunyuan3DPaintConfig(
            view_size,
            camera_config["selected_camera_azims"],
            camera_config["selected_camera_elevs"],
            camera_config["selected_view_weights"],
            camera_config["ortho_scale"],
            texture_size,
        )

        paint_pipeline = Hunyuan3DPaintPipeline(conf)

        image = tensor2pil(image)

        temp_folder_path = os.path.join(script_directory, "temp")
        os.makedirs(temp_folder_path, exist_ok=True)
        temp_output_path = os.path.join(temp_folder_path, "textured_mesh.obj")

        albedo, mr, normal_maps, position_maps = paint_pipeline(
            mesh=trimesh,
            image_path=image,
            output_mesh_path=temp_output_path,
            num_steps=steps,
            guidance_scale=guidance_scale,
            unwrap=unwrap_mesh,
            seed=seed,
        )

        albedo_tensor = hy3dpaintimages_to_tensor(albedo)
        mr_tensor = hy3dpaintimages_to_tensor(mr)
        normals_tensor = hy3dpaintimages_to_tensor(normal_maps)
        positions_tensor = hy3dpaintimages_to_tensor(position_maps)

        return (
            paint_pipeline,
            albedo_tensor,
            mr_tensor,
            positions_tensor,
            normals_tensor,
            camera_config,
        )


class Hy3DBakeMultiViews:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HY3DPIPELINE",),
                "camera_config": ("HY3D21CAMERA",),
                "albedo": ("IMAGE",),
                "mr": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "HY3DPIPELINE",
        "NPARRAY",
        "NPARRAY",
        "NPARRAY",
        "NPARRAY",
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "pipeline",
        "albedo",
        "albedo_mask",
        "mr",
        "mr_mask",
        "albedo_texture",
        "mr_texture",
    )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, pipeline, camera_config, albedo, mr):
        albedo = convert_tensor_images_to_pil(albedo)
        mr = convert_tensor_images_to_pil(mr)

        texture, mask, texture_mr, mask_mr = pipeline.bake_from_multiview(
            albedo,
            mr,
            camera_config["selected_camera_elevs"],
            camera_config["selected_camera_azims"],
            camera_config["selected_view_weights"],
        )

        texture_pil = convert_ndarray_to_pil(texture)
        # mask_pil = convert_ndarray_to_pil(mask)
        texture_mr_pil = convert_ndarray_to_pil(texture_mr)
        # mask_mr_pil = convert_ndarray_to_pil(mask_mr)

        texture_tensor = pil2tensor(texture_pil)
        # mask_tensor = pil2tensor(mask_pil)
        texture_mr_tensor = pil2tensor(texture_mr_pil)
        # mask_mr_tensor = pil2tensor(mask_mr_pil)

        return (
            pipeline,
            texture,
            mask,
            texture_mr,
            mask_mr,
            texture_tensor,
            texture_mr_tensor,
        )


class Hy3DInPaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HY3DPIPELINE",),
                "albedo": ("NPARRAY",),
                "albedo_mask": ("NPARRAY",),
                "mr": ("NPARRAY",),
                "mr_mask": ("NPARRAY",),
                "output_mesh_name": ("STRING",),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "TRIMESH",
        "STRING",
    )
    RETURN_NAMES = ("albedo", "mr", "trimesh", "output_glb_path")
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"
    OUTPUT_NODE = True

    def process(self, pipeline, albedo, albedo_mask, mr, mr_mask, output_mesh_name):

        # albedo = tensor2pil(albedo)
        # albedo_mask = tensor2pil(albedo_mask)
        # mr = tensor2pil(mr)
        # mr_mask = tensor2pil(mr_mask)

        vertex_inpaint = True
        method = "NS"

        albedo, mr = pipeline.inpaint(
            albedo, albedo_mask, mr, mr_mask, vertex_inpaint, method
        )

        pipeline.set_texture_albedo(albedo)
        pipeline.set_texture_mr(mr)

        temp_folder_path = os.path.join(script_directory, "temp")
        os.makedirs(temp_folder_path, exist_ok=True)
        output_mesh_path = os.path.join(temp_folder_path, f"{output_mesh_name}.obj")
        output_temp_path = pipeline.save_mesh(output_mesh_path)

        output_glb_path = os.path.join(script_directory, "output", "3D", f"{output_mesh_name}.glb")
        os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
        shutil.copyfile(output_temp_path, output_glb_path)

        trimesh = Trimesh.load(output_glb_path, force="mesh")

        texture_pil = convert_ndarray_to_pil(albedo)
        texture_mr_pil = convert_ndarray_to_pil(mr)
        texture_tensor = pil2tensor(texture_pil)
        texture_mr_tensor = pil2tensor(texture_mr_pil)

        output_glb_path = f"{output_mesh_name}.glb"

        pipeline.clean_memory()

        del pipeline

        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

        return (texture_tensor, texture_mr_tensor, trimesh, output_glb_path)


class Hy3D21CameraConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_azimuths": (
                    "STRING",
                    {"default": "0, 90, 180, 270, 0, 180", "multiline": False},
                ),
                "camera_elevations": (
                    "STRING",
                    {"default": "0, 0, 0, 0, 90, -90", "multiline": False},
                ),
                "view_weights": (
                    "STRING",
                    {"default": "1, 0.1, 0.5, 0.1, 0.05, 0.05", "multiline": False},
                ),
                "ortho_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("HY3D21CAMERA",)
    RETURN_NAMES = ("camera_config",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, camera_azimuths, camera_elevations, view_weights, ortho_scale):
        angles_list = list(map(int, camera_azimuths.replace(" ", "").split(",")))
        elevations_list = list(map(int, camera_elevations.replace(" ", "").split(",")))
        weights_list = list(map(float, view_weights.replace(" ", "").split(",")))

        camera_config = {
            "selected_camera_azims": angles_list,
            "selected_camera_elevs": elevations_list,
            "selected_view_weights": weights_list,
            "ortho_scale": ortho_scale,
        }

        return (camera_config,)


class Hy3D21VAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("vae"),
                    {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"},
                ),
            },
            "optional": {
                "vae_config": ("HY3D21VAECONFIG",),
            },
        }

    RETURN_TYPES = ("HY3DVAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3D21Wrapper"

    def loadmodel(self, model_name, vae_config=None):
        model_path = folder_paths.get_full_path("vae", model_name)

        vae_sd = load_torch_file(model_path)
        
        # Ensure vae_sd is a dictionary, not a tuple
        if isinstance(vae_sd, tuple):
            vae_sd = vae_sd[0]

        if vae_config is None:
            vae_config = {
                "num_latents": 4096,
                "embed_dim": 64,
                "num_freqs": 8,
                "include_pi": False,
                "heads": 16,
                "width": 1024,
                "num_encoder_layers": 8,
                "num_decoder_layers": 16,
                "qkv_bias": False,
                "qk_norm": True,
                "scale_factor": 1.0039506158752403,
                "geo_decoder_mlp_expand_ratio": 4,
                "geo_decoder_downsample_ratio": 1,
                "geo_decoder_ln_post": True,
                "point_feats": 4,
                "pc_size": 81920,
                "pc_sharpedge_size": 0,
            }
        if not isinstance(vae_config, dict):
            raise ValueError("vae_config must be a dict with string keys")
        vae = ShapeVAE(**vae_config)
        vae.load_state_dict(vae_sd)
        vae.eval().to(torch.float16)

        return (vae,)


class Hy3D21VAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("HY3DVAE",),
                "latents": ("HY3DLATENT",),
                "box_v": (
                    "FLOAT",
                    {"default": 1.01, "min": -10.0, "max": 10.0, "step": 0.001},
                ),
                "octree_resolution": (
                    "INT",
                    {"default": 384, "min": 8, "max": 4096, "step": 8},
                ),
                "num_chunks": (
                    "INT",
                    {
                        "default": 8000,
                        "min": 1,
                        "max": 10000000,
                        "step": 1,
                        "tooltip": "Number of chunks to process at once, higher values use more memory, but make the process faster",
                    },
                ),
                "mc_level": (
                    "FLOAT",
                    {"default": 0, "min": -1.0, "max": 1.0, "step": 0.0001},
                ),
                "mc_algo": (["mc", "dmc"], {"default": "mc"}),
            },
            "optional": {
                "enable_flash_vdm": ("BOOLEAN", {"default": True}),
                "force_offload": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Offloads the model to the offload device once the process is done.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(
        self,
        vae,
        latents,
        box_v,
        octree_resolution,
        mc_level,
        num_chunks,
        mc_algo,
        enable_flash_vdm,
        force_offload,
    ):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.soft_empty_cache()
        torch.cuda.empty_cache()

        vae.to(device)

        vae.enable_flashvdm_decoder(enabled=enable_flash_vdm, mc_algo=mc_algo)

        latents = latents.clone().detach()
        latents = vae.decode(latents)
        outputs = vae.latents2mesh(
            latents,
            output_type="trimesh",
            bounds=box_v,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
            enable_pbar=True,
        )[0]

        if force_offload:
            vae.to(offload_device)

        outputs.mesh_f = outputs.mesh_f[:, ::-1]
        mesh_output = Trimesh.Trimesh(outputs.mesh_v, outputs.mesh_f)
        print(
            f"Decoded mesh with {mesh_output.vertices.shape[0]} vertices and {mesh_output.faces.shape[0]} faces"
        )

        # del pipeline
        del vae

        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

        return (mesh_output,)


class Hy3D21PostprocessMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "remove_floaters": ("BOOLEAN", {"default": True}),
                "remove_degenerate_faces": ("BOOLEAN", {"default": True}),
                "reduce_faces": ("BOOLEAN", {"default": True}),
                "max_facenum": (
                    "INT",
                    {"default": 40000, "min": 1, "max": 10000000, "step": 1},
                ),
                "smooth_normals": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(
        self,
        trimesh,
        remove_floaters,
        remove_degenerate_faces,
        reduce_faces,
        max_facenum,
        smooth_normals,
    ):
        new_mesh = trimesh.copy()
        if remove_floaters:
            new_mesh = FloaterRemover()(new_mesh)
            print(
                f"Removed floaters, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces"
            )
        if remove_degenerate_faces:
            new_mesh = DegenerateFaceRemover()(new_mesh)
            print(
                f"Removed degenerate faces, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces"
            )
        if reduce_faces:
            new_mesh = FaceReducer()(new_mesh, max_facenum=max_facenum)
            print(
                f"Reduced faces, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces"
            )
        if smooth_normals:
            new_mesh.vertex_normals = Trimesh.smoothing.get_vertices_normals(new_mesh)

        return (new_mesh,)


class Hy3D21ExportMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/Hy3D"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
            "optional": {
                "save_file": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, filename_prefix, file_format, save_file=True):
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
        )
        output_glb_path = Path(
            full_output_folder, f"{filename}_{counter:05}_.{file_format}"
        )
        output_glb_path.parent.mkdir(exist_ok=True)
        if save_file:
            trimesh.export(output_glb_path, file_type=file_format)
            relative_path = Path(subfolder) / f"{filename}_{counter:05}_.{file_format}"
        else:
            temp_file = Path(full_output_folder, f"hy3dtemp_.{file_format}")
            trimesh.export(temp_file, file_type=file_format)
            relative_path = Path(subfolder) / f"hy3dtemp_.{file_format}"

        return (str(relative_path),)


class Hy3D21MeshUVWrap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3D21Wrapper"

    def process(self, trimesh):
        trimesh = mesh_uv_wrap(trimesh)

        return (trimesh,)


# class Hy3D21LoadMesh:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "glb_path": (
#                     "STRING",
#                     {"default": "", "tooltip": "The glb path with mesh to load."},
#                 ),
#             }
#         }

#     RETURN_TYPES = ("TRIMESH",)
#     RETURN_NAMES = ("trimesh",)
#     OUTPUT_TOOLTIPS = ("The glb model with mesh to texturize.",)

#     FUNCTION = "load"
#     CATEGORY = "Hunyuan3D21Wrapper"
#     DESCRIPTION = "Loads a glb model from the given path."

#     def load(self, glb_path):

#         if not os.path.exists(glb_path):
#             glb_path = os.path.join(folder_paths.get_input_directory(), glb_path)

#         trimesh = Trimesh.load(glb_path, force="mesh")

#         return (trimesh,)


class Hy3D21MeshlibDecimate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "subdivideParts": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Should be the number of CPU/Core",
                    },
                ),
            },
            "optional": {
                "target_face_num": ("INT", {"min": 0, "max": 10000000}),
                "target_face_ratio": ("FLOAT", {"min": 0.000, "max": 0.999}),
                "strategy": (
                    ["None", "MinimizeError", "ShortestEdgeFirst"],
                    {"default": "None"},
                ),
                "maxError": ("FLOAT", {"min": 0.0, "max": 1.0}),
                "maxEdgeLen": ("FLOAT",),
                "maxBdShift": ("FLOAT",),
                "maxTriangleAspectRatio": ("FLOAT",),
                "criticalTriAspectRatio": ("FLOAT",),
                "tinyEdgeLength": ("FLOAT",),
                "stabilizer": ("FLOAT",),
                "angleWeightedDistToPlane": ("BOOLEAN",),
                "optimizeVertexPos": ("BOOLEAN",),
                "collapseNearNotFlippable": ("BOOLEAN",),
                "touchNearBdEdges": ("BOOLEAN",),
                "maxAngleChange": ("FLOAT",),
                "decimateBetweenParts": ("BOOLEAN",),
                "minFacesInPart": ("INT",),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "decimate"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Decimate the mesh using meshlib: https://meshlib.io/"

    def decimate(
        self,
        trimesh,
        subdivideParts,
        target_face_num=0,
        target_face_ratio=0.0,
        strategy="None",
        maxError=0.0,
        maxEdgeLen=0.0,
        maxBdShift=0.0,
        maxTriangleAspectRatio=0.0,
        criticalTriAspectRatio=0.0,
        tinyEdgeLength=0.0,
        stabilizer=0.0,
        angleWeightedDistToPlane=False,
        optimizeVertexPos=False,
        collapseNearNotFlippable=False,
        touchNearBdEdges=False,
        maxAngleChange=0.0,
        decimateBetweenParts=False,
        minFacesInPart=0,
    ):
        if target_face_num == 0 and target_face_ratio == 0.0:
            raise ValueError("target_face_num or target_face_ratio must be set")

        current_faces_num = trimesh.faces.shape[0]
        print(f"Current Faces Number: {current_faces_num}")

        settings = mrmeshpy.DecimateSettings()
        if target_face_num > 0:
            faces_to_delete = current_faces_num - target_face_num
            settings.maxDeletedFaces = faces_to_delete
        elif target_face_ratio > 0.0:
            target_faces = int(current_faces_num * target_face_ratio)
            faces_to_delete = current_faces_num - target_faces
            settings.maxDeletedFaces = faces_to_delete
        else:
            raise ValueError("target_face_num or target_face_ratio must be set")

        if strategy == "MinimizeError":
            settings.strategy = mrmeshpy.DecimateStrategy.MinimizeError
        elif strategy == "ShortestEdgeFirst":
            settings.strategy = mrmeshpy.DecimateStrategy.ShortestEdgeFirst

        if maxError > 0.0:
            settings.maxError = maxError
        if maxEdgeLen > 0.0:
            settings.maxEdgeLen = maxEdgeLen
        if maxBdShift > 0.0:
            settings.maxBdShift = maxBdShift
        if maxTriangleAspectRatio > 0.0:
            settings.maxTriangleAspectRatio = maxTriangleAspectRatio
        if criticalTriAspectRatio > 0.0:
            settings.criticalTriAspectRatio = criticalTriAspectRatio
        if tinyEdgeLength > 0.0:
            settings.tinyEdgeLength = tinyEdgeLength
        if stabilizer > 0.0:
            settings.stabilizer = stabilizer
        if angleWeightedDistToPlane:
            settings.angleWeightedDistToPlane = angleWeightedDistToPlane
        if optimizeVertexPos:
            settings.optimizeVertexPos = optimizeVertexPos
        if collapseNearNotFlippable:
            settings.collapseNearNotFlippable = collapseNearNotFlippable
        if touchNearBdEdges:
            settings.touchNearBdEdges = touchNearBdEdges
        if maxAngleChange > 0.0:
            settings.maxAngleChange = maxAngleChange
        if decimateBetweenParts:
            settings.decimateBetweenParts = decimateBetweenParts
        if minFacesInPart > 0:
            settings.minFacesInPart = minFacesInPart

        settings.packMesh = True
        settings.subdivideParts = subdivideParts

        new_mesh = postprocessmesh(trimesh.vertices, trimesh.faces, settings)

        return (new_mesh,)


class Hy3D21SimpleMeshlibDecimate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "subdivideParts": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Should be the number of CPU/Core",
                    },
                ),
            },
            "optional": {
                "target_face_num": ("INT", {"min": 0, "max": 10000000}),
                "target_face_ratio": ("FLOAT", {"min": 0.000, "max": 0.999}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "decimate"
    CATEGORY = "Hunyuan3D21Wrapper"
    DESCRIPTION = "Decimate the mesh using meshlib: https://meshlib.io/"

    def decimate(
        self, trimesh, subdivideParts, target_face_num=0, target_face_ratio=0.0
    ):
        if target_face_num == 0 and target_face_ratio == 0.0:
            raise ValueError("target_face_num or target_face_ratio must be set")

        current_faces_num = trimesh.faces.shape[0]
        print(f"Current Faces Number: {current_faces_num}")

        settings = mrmeshpy.DecimateSettings()
        if target_face_num > 0:
            faces_to_delete = current_faces_num - target_face_num
            settings.maxDeletedFaces = faces_to_delete
        elif target_face_ratio > 0.0:
            target_faces = int(current_faces_num * target_face_ratio)
            faces_to_delete = current_faces_num - target_faces
            settings.maxDeletedFaces = faces_to_delete
        else:
            raise ValueError("target_face_num or target_face_ratio must be set")

        settings.packMesh = True
        settings.subdivideParts = subdivideParts

        new_mesh = postprocessmesh(trimesh.vertices, trimesh.faces, settings)

        return (new_mesh,)


class Hy3D21UpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("upscale_models"),
                    {"tooltip": "These models are loaded from 'models/upscale_models'"},
                ),
            },
        }

    RETURN_TYPES = ("UPSCALE_MODEL",)
    RETURN_NAMES = ("upscale_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3D21Wrapper"

    def loadmodel(self, model_name):
        model_path = folder_paths.get_full_path("upscale_models", model_name)

        # Determine model type and scale based on filename
        if "x4plus" in model_name.lower():
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            scale = 4
        elif "x2plus" in model_name.lower():
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            scale = 2
        elif "x8" in model_name.lower():
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=8,
            )
            scale = 8
        else:
            # Default to x4
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            scale = 4

        # Create RealESRGAN upsampler
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            gpu_id=None,
        )

        return (upsampler,)


class Hy3D21UpscaleImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "Hunyuan3D21Wrapper"

    def upscale(self, upscale_model, image):
        # Convert tensor to PIL for processing
        pil_image = tensor2pil(image.squeeze(0))

        # Convert PIL to numpy array (BGR format for cv2)
        img_np = np.array(pil_image)
        if img_np.shape[2] == 3:  # RGB to BGR
            img_np = cvtColor(img_np, COLOR_RGB2BGR)

        # Upscale the image
        try:
            output, _ = upscale_model.enhance(img_np, outscale=upscale_model.scale)
        except Exception as e:
            print(f"Upscaling failed: {e}")
            # Fallback to original image
            output = img_np

        # Convert back to RGB
        if output.shape[2] == 3:
            output = cvtColor(output, COLOR_BGR2RGB)

        # Convert to PIL and then to tensor
        output_pil = Image.fromarray(output)
        upscaled_tensor = pil2tensor(output_pil)

        return (upscaled_tensor,)


# class Hy3DHighPolyToLowPolyBakeMultiViewsWithMetaData:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "metadata_file": ("STRING",),
#                 "view_size": ("INT", {"default": 512}),
#                 "texture_size": ("INT", {"default": 1024}),
#                 "target_face_nums": ("STRING", {"default": "20000,10000,5000"}),
#             },
#         }

#     RETURN_TYPES = ("STRING",)
#     RETURN_NAMES = ("output_lowpoly_path",)
#     FUNCTION = "process"
#     CATEGORY = "Hunyuan3D21Wrapper"
#     OUTPUT_NODE = True

#     def process(self, metadata_file, view_size, texture_size, target_face_nums):
#         output_lowpoly_path = ""

#         vertex_inpaint = True
#         method = "NS"

#         with open(metadata_file, "r") as fr:
#             loaded_data = json.load(fr)
#             loaded_metaData = MetaData()
#             for key, value in loaded_data.items():
#                 setattr(loaded_metaData, key, value)

#         list_of_faces = parse_string_to_int_list(target_face_nums)
#         if len(list_of_faces) > 0:
#             input_dir = os.path.dirname(metadata_file)
#             mesh_name = loaded_metaData.mesh_file.replace(".glb", "").replace(
#                 ".obj", ""
#             )
#             mesh_file_path = os.path.join(input_dir, loaded_metaData.mesh_file)

#             if os.path.exists(mesh_file_path):
#                 conf = Hunyuan3DPaintConfig(
#                     view_size,
#                     loaded_metaData.camera_config["selected_camera_azims"],
#                     loaded_metaData.camera_config["selected_camera_elevs"],
#                     loaded_metaData.camera_config["selected_view_weights"],
#                     loaded_metaData.camera_config["ortho_scale"],
#                     texture_size,
#                 )

#                 highpoly_mesh = Trimesh.load(mesh_file_path, force="mesh")
#                 highpoly_mesh = Trimesh.Trimesh(
#                     vertices=highpoly_mesh.vertices, faces=highpoly_mesh.faces
#                 )  # Remove texture coordinates
#                 highpoly_faces_num = highpoly_mesh.faces.shape[0]

#                 albedos = []
#                 mrs = []

#                 if loaded_metaData.albedos_upscaled != None:
#                     print("Using upscaled pictures ...")
#                     for file in loaded_metaData.albedos_upscaled:
#                         albedo_file = os.path.join(input_dir, file)
#                         albedo = Image.open(albedo_file)
#                         albedos.append(albedo)

#                     for file in loaded_metaData.mrs_upscaled:
#                         mr_file = os.path.join(input_dir, file)
#                         mr = Image.open(mr_file)
#                         mrs.append(mr)
#                 else:
#                     print("Using non-upscaled pictures ...")
#                     for file in loaded_metaData.albedos:
#                         albedo_file = os.path.join(input_dir, file)
#                         albedo = Image.open(albedo_file)
#                         albedos.append(albedo)

#                     for file in loaded_metaData.mrs:
#                         mr_file = os.path.join(dir_name, file)
#                         mr = Image.open(mr_file)
#                         mrs.append(mr)

#                 output_lowpoly_path = os.path.join(input_dir, "LowPoly")

#                 for target_face_num in list_of_faces:
#                     print("Processing {target_face_num} faces ...")
#                     pipeline = Hunyuan3DPaintPipeline(conf)
#                     output_dir_path = os.path.join(
#                         input_dir, "LowPoly", f"{target_face_num}"
#                     )
#                     os.makedirs(output_dir_path, exist_ok=True)

#                     settings = mrmeshpy.DecimateSettings()
#                     faces_to_delete = highpoly_faces_num - target_face_num
#                     settings.maxDeletedFaces = faces_to_delete
#                     settings.subdivideParts = 16
#                     settings.packMesh = True

#                     print(f"Decimating to {target_face_num} faces ...")
#                     lowpoly_mesh = postprocessmesh(
#                         highpoly_mesh.vertices, highpoly_mesh.faces, settings
#                     )

#                     print("UV Unwrapping ...")
#                     lowpoly_mesh = mesh_uv_wrap(lowpoly_mesh)

#                     pipeline.load_mesh(lowpoly_mesh)

#                     camera_config = loaded_metaData.camera_config
#                     texture, mask, texture_mr, mask_mr = pipeline.bake_from_multiview(
#                         albedos,
#                         mrs,
#                         camera_config["selected_camera_elevs"],
#                         camera_config["selected_camera_azims"],
#                         camera_config["selected_view_weights"],
#                     )

#                     albedo, mr = pipeline.inpaint(
#                         texture, mask, texture_mr, mask_mr, vertex_inpaint, method
#                     )

#                     pipeline.set_texture_albedo(albedo)
#                     pipeline.set_texture_mr(mr)

#                     output_glb_path = os.path.join(
#                         output_dir_path, f"{mesh_name}_{target_face_num}.obj"
#                     )

#                     pipeline.save_mesh(output_glb_path)

#                     pipeline.clean_memory()

#             else:
#                 print(f"Mesh file does not exist: {mesh_file_path}")
#         else:
#             print("target_face_nums is empty")

#         return (output_lowpoly_path,)


NODE_CLASS_MAPPINGS = {
    "Hy3DMeshGenerator": Hy3DMeshGenerator,
    "Hy3DMultiViewsGenerator": Hy3DMultiViewsGenerator,
    "Hy3DBakeMultiViews": Hy3DBakeMultiViews,
    "Hy3DInPaint": Hy3DInPaint,
    "Hy3D21CameraConfig": Hy3D21CameraConfig,
    "Hy3D21VAELoader": Hy3D21VAELoader,
    "Hy3D21VAEDecode": Hy3D21VAEDecode,
    "Hy3D21PostprocessMesh": Hy3D21PostprocessMesh,
    "Hy3D21ExportMesh": Hy3D21ExportMesh,
    "Hy3D21MeshUVWrap": Hy3D21MeshUVWrap,
    # "Hy3D21LoadMesh": Hy3D21LoadMesh,
    "Hy3D21MeshlibDecimate": Hy3D21MeshlibDecimate,
    # "Hy3DHighPolyToLowPolyBakeMultiViewsWithMetaData": Hy3DHighPolyToLowPolyBakeMultiViewsWithMetaData,
    "Hy3D21SimpleMeshlibDecimate": Hy3D21SimpleMeshlibDecimate,
    "Hy3D21UpscaleModelLoader": Hy3D21UpscaleModelLoader,
    "Hy3D21UpscaleImage": Hy3D21UpscaleImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hy3DMeshGenerator": "Hunyuan 3D 2.1 Mesh Generator",  # This
    "Hy3DMultiViewsGenerator": "Hunyuan 3D 2.1 MultiViews Generator",  # This
    "Hy3DBakeMultiViews": "Hunyuan 3D 2.1 Bake MultiViews",  # This
    "Hy3DInPaint": "Hunyuan 3D 2.1 InPaint",  # This
    "Hy3D21CameraConfig": "Hunyuan 3D 2.1 Camera Config",  # This
    "Hy3D21VAELoader": "Hunyuan 3D 2.1 VAE Loader",  # This
    "Hy3D21VAEDecode": "Hunyuan 3D 2.1 VAE Decoder",  # This
    "Hy3D21PostprocessMesh": "Hunyuan 3D 2.1 Post Process Trimesh",  # This
    "Hy3D21ExportMesh": "Hunyuan 3D 2.1 Export Mesh",  # This
    "Hy3D21MeshUVWrap": "Hunyuan 3D 2.1 Mesh UV Wrap",  # This
    "Hy3D21UpscaleModelLoader": "Hunyuan 3D 2.1 Upscale Model Loader",
    "Hy3D21UpscaleImage": "Hunyuan 3D 2.1 Upscale Image",
    # "Hy3D21LoadMesh": "Hunyuan 3D 2.1 Load Mesh",
    # "Hy3D21MeshlibDecimate": "Hunyuan 3D 2.1 Meshlib Decimation",
    # "Hy3DHighPolyToLowPolyBakeMultiViewsWithMetaData": "Hunyuan 3D 2.1 HighPoly to LowPoly Bake MultiViews With MetaData",
    # "Hy3D21SimpleMeshlibDecimate": "Hunyuan 3D 2.1 Simple Meshlib Decimation"
}
