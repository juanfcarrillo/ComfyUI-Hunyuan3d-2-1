"""
Hunyuan 3D 2.1 Manual Workflow
=============================

This script provides multiple workflow classes for generating 3D models with textures
from 2D images using the Hunyuan 3D 2.1 model.

WORKFLOWS AVAILABLE:
1. ManualHunyuan3DWorkflow - Mesh generation only
2. ManualHunyuan3DTextureWorkflow - Texture generation for existing meshes
3. CompleteHunyuan3DWorkflow - Full pipeline (mesh + texture)
4. EnhancedHunyuan3DWorkflow - **RECOMMENDED** (mesh + texture + decimation + full camera control)

WORKFLOW COMPARISON:
==================

ManualHunyuan3DWorkflow:
- Generates 3D mesh from 2D image
- Optimized for 16GB VRAM
- Fastest option
- Output: Single GLB file in output/3D/

ManualHunyuan3DTextureWorkflow:
- Applies textures to existing 3D mesh
- Requires pre-existing mesh
- Full camera configuration control
- Output: Textured GLB file in output/3D/

CompleteHunyuan3DWorkflow:
- Combines mesh generation + texture generation
- Uses default camera settings (6 views)
- Moderate memory usage
- Output: Base mesh + textured mesh in output/3D/

EnhancedHunyuan3DWorkflow (RECOMMENDED):
- All features of Complete workflow
- PLUS automatic mesh decimation for optimal performance
- PLUS full camera configuration control
- PLUS advanced memory management
- PLUS customizable texture parameters
- PLUS optional background removal with INSPYRENET/rembg fallback
- Best for 16GB VRAM systems
- Output: Base mesh + decimated mesh + final textured mesh in output/3D/

CAMERA CONFIGURATION OPTIONS (Enhanced Workflow):
- camera_azimuths: Camera angles around object (comma-separated)
- camera_elevations: Camera height angles (comma-separated)
- view_weights: Importance weights for each view (comma-separated)
- ortho_scale: Orthographic camera scale
- view_size: Render resolution per view (affects quality vs speed)
- texture_size: Final texture resolution
- steps: Diffusion steps for texture generation
- guidance_scale: How closely to follow the input image

OPTIMIZATIONS FOR 16GB VRAM:
- Reduced view_size from 1024 to 512
- Reduced texture_size from 4096 to 1024-2048
- Reduced octree_resolution from 384 to 224-256
- Reduced num_chunks from 8000 to 3000-4000
- Reduced steps and guidance_scale
- Enabled flash_vdm and force_offload
- Added mesh decimation to reduce face count
- Disabled texture upscaling by default

USAGE EXAMPLES:
# Basic mesh generation
python manual_workflow.py --workflow mesh --input-image assets/mune.png

# Basic mesh generation with background removal
python manual_workflow.py --workflow mesh --input-image assets/mune.png --remove-background

# Complete pipeline (mesh + texture)
python manual_workflow.py --workflow complete --input-image assets/mune.png

# Complete pipeline with background removal
python manual_workflow.py --workflow complete --input-image assets/mune.png --remove-background

# Enhanced pipeline (recommended for 16GB VRAM)
python manual_workflow.py --workflow enhanced --input-image assets/mune.png

# Enhanced pipeline with background removal
python manual_workflow.py --workflow enhanced --input-image assets/mune.png --remove-background

# Custom background removal model path
python manual_workflow.py --workflow enhanced --input-image assets/mune.png --remove-background --bg-model-path "models/RMBG/INSPYRENET/inspyrenet.safetensors"

REQUIREMENTS:
- All dependencies from requirements.txt must be installed
- Models should be placed in the appropriate directories
- Virtual environment should be activated before running
- For background removal: insightface package for INSPYRENET (optional, falls back to rembg)
"""

import torch
import gc
import logging
from PIL import Image
import numpy as np
import os

# Import the node classes from the nodes.py file
from nodes import (
    Hy3D21VAELoader,
    Hy3DMeshGenerator,
    Hy3D21VAEDecode,
    Hy3D21PostprocessMesh,
    Hy3D21ExportMesh,
    Hy3D21MeshUVWrap,
    Hy3D21CameraConfig,
    Hy3DMultiViewsGenerator,
    Hy3DBakeMultiViews,
    Hy3DInPaint,
    Hy3D21UpscaleModelLoader,
    Hy3D21UpscaleImage,
    Hy3D21SimpleMeshlibDecimate,
    Hy3D21MeshlibDecimate,
    # Utility functions
    pil2tensor,
    tensor2pil,
    hy3dpaintimages_to_tensor,
    convert_ndarray_to_pil,
    convert_tensor_images_to_pil,
)

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Background removal imports
try:
    from hy3dshape.hy3dshape.rembg import BackgroundRemover as RembgRemover
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class BackgroundRemover:
    """Enhanced background remover using ComfyUI InSPyReNet custom node"""

    def __init__(self, model_path=None, threshold=0.5, use_jit=False):
        self.model_path = model_path or "models/RMBG/INSPYRENET/inspyrenet.safetensors"
        self.threshold = threshold
        self.use_jit = use_jit
        self.inspyrenet_available = False

        # Check if the ComfyUI InSPyReNet custom node is available
        try:
            from transparent_background import Remover
            self.remover = Remover(jit=self.use_jit)
            self.inspyrenet_available = True
            print("✅ InSPyReNet background removal initialized via transparent-background library")
            print(f"   Threshold: {self.threshold}, JIT: {self.use_jit}")
            print("   This uses the actual InSPyReNet model architecture for superior quality")
        except ImportError:
            print("❌ transparent-background library not available")
            print("   Please install ComfyUI-Inspyrenet-Rembg custom node")
            self.inspyrenet_available = False
        except Exception as e:
            print(f"❌ Failed to initialize InSPyReNet: {e}")
            self.inspyrenet_available = False

    def remove_background_inspyrenet(self, image, threshold=None):
        """Remove background using actual InSPyReNet model"""
        try:
            if not self.inspyrenet_available:
                print("InSPyReNet not available")
                return None

            # Use provided threshold or default
            thresh = threshold if threshold is not None else self.threshold

            print(f"🎯 Processing image with InSPyReNet (threshold: {thresh})...")

            # Convert to RGBA for transparency
            result = self.remover.process(image, type='rgba', threshold=thresh)
            print("✅ InSPyReNet background removal completed successfully")
            return result

        except Exception as e:
            print(f"❌ InSPyReNet background removal failed: {e}")
            return None

    def remove_background(self, image, threshold=None):
        """Remove background from image using InSPyReNet"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if self.inspyrenet_available:
            print("🎯 Using actual InSPyReNet model for background removal...")
            result = self.remove_background_inspyrenet(image, threshold)
            if result is not None:
                return result

        # If InSPyReNet fails, return original image
        print("❌ InSPyReNet background removal failed, returning original image")
        return image

    def update_settings(self, threshold=None, use_jit=None):
        """Update background removal settings"""
        if threshold is not None:
            self.threshold = threshold
        if use_jit is not None:
            self.use_jit = use_jit
            # Reinitialize remover with new JIT setting
            if self.inspyrenet_available:
                try:
                    from transparent_background import Remover
                    self.remover = Remover(jit=self.use_jit)
                    print(f"✅ Remover updated with JIT: {self.use_jit}")
                except Exception as e:
                    print(f"❌ Failed to update remover: {e}")


def cleanup_memory():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


class ManualHunyuan3DWorkflow:
    def __init__(self):
        # Initialize the node instances
        self.vae_loader = Hy3D21VAELoader()
        self.mesh_generator = Hy3DMeshGenerator()
        self.vae_decoder = Hy3D21VAEDecode()
        self.postprocess_mesh = Hy3D21PostprocessMesh()
        self.export_mesh = Hy3D21ExportMesh()
        # Add mesh decimation nodes
        self.simple_decimate = Hy3D21SimpleMeshlibDecimate()
        self.advanced_decimate = Hy3D21MeshlibDecimate()

    def load_image(self, image_path, remove_background=False, bg_model_path=None, bg_threshold=0.5, bg_use_jit=False):
        """Load image and convert to tensor format expected by the nodes"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path

        # Optional background removal
        if remove_background:
            print("Removing background from input image...")
            bg_remover = BackgroundRemover(bg_model_path, threshold=bg_threshold, use_jit=bg_use_jit)
            image = bg_remover.remove_background(image)
            print("Background removal completed")

        # Convert PIL to tensor format (add batch dimension)
        image_tensor = pil2tensor(image)
        return image_tensor

    def get_number_of_faces(self, trimesh):
        """Get the number of faces from a trimesh object"""
        return trimesh.faces.shape[0]

    def decimate_mesh(self, trimesh, target_faces=15000, method="simple"):
        """Decimate mesh to reduce face count for better performance"""
        print(f"   Decimating mesh from {self.get_number_of_faces(trimesh)} to ~{target_faces} faces...")

        if method == "simple":
            decimated = self.simple_decimate.decimate(
                trimesh=trimesh,
                subdivideParts=8,  # Use 8 CPU cores
                target_face_num=target_faces,
            )[0]
        else:
            decimated = self.advanced_decimate.decimate(
                trimesh=trimesh,
                subdivideParts=8,  # Use 8 CPU cores
                target_face_num=target_faces,
                strategy="MinimizeError",  # Better quality
                maxError=0.01,  # Low error tolerance
            )[0]

        print(f"   Decimation complete. Faces: {self.get_number_of_faces(trimesh)} → {self.get_number_of_faces(decimated)}")
        return decimated

    def run_workflow(
        self,
        vae_model_name,
        diffusion_model_name,
        input_image_path,
        output_mesh_name="generated_mesh",
        # VAE Loader parameters
        vae_config=None,
        # Mesh Generator parameters (optimized for 16GB VRAM)
        steps=30,  # Reduced from 50
        guidance_scale=4.0,  # Reduced from 5.0
        seed=0,
        attention_mode="sdpa",  # Memory efficient
        # VAE Decoder parameters (optimized for 16GB VRAM)
        box_v=1.01,
        octree_resolution=256,  # Reduced from 384
        num_chunks=4000,  # Reduced from 8000
        mc_level=0,
        mc_algo="mc",
        enable_flash_vdm=True,  # Memory efficient
        force_offload=True,  # Enable offloading
        # Post Process parameters
        remove_floaters=True,
        remove_degenerate_faces=True,
        reduce_faces=True,
        max_facenum=25000,  # Reduced from 40000 for 16GB VRAM
        smooth_normals=False,
        # Export parameters
        file_format="glb",
        save_file=True,
        # Background removal parameters
        remove_background=False,
        bg_model_path=None,
        bg_threshold=0.5,
        bg_use_jit=False,
    ):

        print("Starting Hunyuan 3D 2.1 Manual Workflow...")

        # Step 1: Load VAE model
        print("1. Loading VAE model...")
        cleanup_memory()  # Clean memory before loading
        vae = self.vae_loader.loadmodel(vae_model_name, vae_config)[0]
        print(f"   VAE model loaded: {vae_model_name}")

        # Step 2: Load and prepare input image
        print("2. Loading input image...")
        image_tensor = self.load_image(input_image_path, remove_background=remove_background, bg_model_path=bg_model_path, bg_threshold=bg_threshold, bg_use_jit=bg_use_jit)
        print(f"   Image loaded from: {input_image_path}")
        print(f"   Image shape: {image_tensor.shape}")

        # Step 3: Generate mesh latents
        print("3. Generating mesh latents...")
        cleanup_memory()  # Clean memory before generation
        latents = self.mesh_generator.loadmodel(
            model=diffusion_model_name,
            image=image_tensor,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            attention_mode=attention_mode,
        )[0]
        print(f"   Latents generated with {steps} steps")
        cleanup_memory()  # Clean up after generation

        # Step 4: Decode latents to mesh
        print("4. Decoding latents to mesh...")
        cleanup_memory()  # Clean memory before decoding
        trimesh = self.vae_decoder.process(
            vae=vae,
            latents=latents,
            box_v=box_v,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            mc_level=mc_level,
            mc_algo=mc_algo,
            enable_flash_vdm=enable_flash_vdm,
            force_offload=force_offload,
        )[0]
        print(
            f"   Mesh decoded with {trimesh.vertices.shape[0]} vertices and {trimesh.faces.shape[0]} faces"
        )
        cleanup_memory()  # Clean up after decoding

        # Step 5: Get number of faces (equivalent to Get_NumberOfFaces node)
        num_faces_before = self.get_number_of_faces(trimesh)
        print(f"5. Number of faces before post-processing: {num_faces_before}")

        # Step 6: Post-process mesh
        print("6. Post-processing mesh...")
        processed_trimesh = self.postprocess_mesh.process(
            trimesh=trimesh,
            remove_floaters=remove_floaters,
            remove_degenerate_faces=remove_degenerate_faces,
            reduce_faces=reduce_faces,
            max_facenum=max_facenum,
            smooth_normals=smooth_normals,
        )[0]

        num_faces_after = self.get_number_of_faces(processed_trimesh)
        print(
            f"   Post-processing complete. Faces: {num_faces_before} → {num_faces_after}"
        )

        # Optional: Decimate mesh for better texture generation performance
        # Use 80% of max_facenum as target if mesh is overly complex
        decimation_threshold = max_facenum * 1.2  # Decimate if faces exceed 120% of max_facenum
        target_faces = int(max_facenum * 0.8)  # Target 80% of max_facenum
        
        if num_faces_after > decimation_threshold:
            print(f"7. Decimating mesh for optimal texture generation...")
            print(f"   Target: {target_faces} faces (80% of max_facenum: {max_facenum})")
            processed_trimesh = self.decimate_mesh(
                processed_trimesh,
                target_faces=target_faces,
                method="simple"
            )
            num_faces_after = self.get_number_of_faces(processed_trimesh)
            print(f"   Final face count: {num_faces_after}")

        # Step 7: Export mesh
        print("8. Exporting mesh...")
        output_path = self.export_mesh.process(
            trimesh=processed_trimesh,
            filename_prefix=f"3D/{output_mesh_name}",
            file_format=file_format,
            save_file=save_file,
        )[0]
        print(f"   Mesh exported to: {output_path}")

        # Return results
        results = {
            "vae": vae,
            "latents": latents,
            "raw_mesh": trimesh,
            "processed_mesh": processed_trimesh,
            "num_faces_before": num_faces_before,
            "num_faces_after": num_faces_after,
            "output_path": output_path,
            "output_mesh_name": output_mesh_name,
            "vram_optimized": octree_resolution < 300,  # Simple heuristic
        }

        print("Mesh generation workflow completed successfully!")
        print(f"Final mesh: {num_faces_after:,} faces")
        print(f"Output: {output_path}")

        return results


class ManualHunyuan3DTextureWorkflow:
    def __init__(self):
        # Initialize the node instances for texture generation
        self.mesh_uv_wrap = Hy3D21MeshUVWrap()
        self.camera_config = Hy3D21CameraConfig()
        self.multiviews_generator = Hy3DMultiViewsGenerator()
        self.bake_multiviews = Hy3DBakeMultiViews()
        self.inpaint = Hy3DInPaint()
        self.upscale_model_loader = Hy3D21UpscaleModelLoader()
        self.upscale_image = Hy3D21UpscaleImage()
        self.export_mesh = Hy3D21ExportMesh()

    def load_image(self, image_path, remove_background=False, bg_model_path=None, bg_threshold=0.5, bg_use_jit=False):
        """Load image and convert to tensor format expected by the nodes"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path

        # Optional background removal
        if remove_background:
            print("Removing background from input image...")
            bg_remover = BackgroundRemover(bg_model_path, threshold=bg_threshold, use_jit=bg_use_jit)
            image = bg_remover.remove_background(image)
            print("Background removal completed")

        # Convert PIL to tensor format (add batch dimension)
        image_tensor = pil2tensor(image)
        return image_tensor

    def run_texture_workflow(
        self,
        input_trimesh,
        input_image_path,
        output_mesh_name="textured_mesh",
        # Camera Config parameters
        camera_azimuths="0, 180, 90, 270, 0, 45",
        camera_elevations="0, 0, 0, 0, -45, -45",
        view_weights="2.0, 2.0, 1.5, 1.5, 0.5, 0.5",
        ortho_scale=1.10,
        # MultiViews Generator parameters (optimized for 16GB VRAM)
        view_size=512,  # Reduced from 1024
        steps=20,  # Reduced from 35
        guidance_scale=4.0,  # Reduced from 5.6
        texture_size=2048,  # Reduced from 4096
        normal_texture=True,
        unwrap_mesh=True,
        save_after_generate=False,
        correct_after_generate="randomize",
        seed=200434251488993,
        # Bake MultiViews parameters
        albedo_texture=True,
        mr_texture=True,
        # InPaint parameters
        vertex_inpaint=True,
        method="NS",
        # Upscale parameters (optimized for 16GB VRAM)
        upscale_model_name="RealESRGAN_x4plus.pth",
        upscale_albedo=True,
        upscale_mr=True,
        # Export parameters
        file_format="glb",
        save_file=True,
        # Background removal parameters
        remove_background=True,
        bg_model_path=None,
        bg_threshold=0.5,
        bg_use_jit=False,
    ):

        print("Starting Hunyuan 3D 2.1 Texture Generation Workflow...")

        # Step 1: Generate UV mapping for mesh
        print("1. Generating UV mapping...")
        uv_wrapped_mesh = self.mesh_uv_wrap.process(
            trimesh=input_trimesh
        )[0]
        print("   UV mapping completed")

        # Step 2: Configure camera settings
        print("2. Configuring camera settings...")
        camera_config = self.camera_config.process(
            camera_azimuths=camera_azimuths,
            camera_elevations=camera_elevations,
            view_weights=view_weights,
            ortho_scale=ortho_scale,
        )[0]
        print(f"   Camera config set with {len(camera_azimuths.split(','))} views")

        # Step 3: Load input image
        print("3. Loading input image...")
        image_tensor = self.load_image(input_image_path, remove_background=remove_background, bg_model_path=bg_model_path, bg_threshold=bg_threshold, bg_use_jit=bg_use_jit)
        print(f"   Image loaded from: {input_image_path}")
        print(f"   Image shape: {image_tensor.shape}")

        # Step 4: Generate multi-views
        print("4. Generating multi-view textures...")
        cleanup_memory()  # Clean memory before multi-view generation
        multiviews_generator = self.multiviews_generator.genmultiviews(
            trimesh=uv_wrapped_mesh,
            camera_config=camera_config,
            view_size=view_size,
            image=image_tensor,
            steps=steps,
            guidance_scale=guidance_scale,
            texture_size=texture_size,
            unwrap_mesh=unwrap_mesh,
            seed=seed,
        )

        # Unpack the correct return values from MultiViewsGenerator (6 values)
        pipeline = multiviews_generator[0]
        albedo = multiviews_generator[1]  # albedo images
        mr = multiviews_generator[2]      # mr images
        positions = multiviews_generator[3]  # position maps
        normals = multiviews_generator[4]    # normal maps
        camera_config_out = multiviews_generator[5]  # camera config
        # Note: metadata is not returned by this version of the node

        print("   Multi-view generation completed")
        print(f"   Generated textures with size {texture_size}x{texture_size}")
        cleanup_memory()  # Clean up after multi-view generation

        # Step 5: Bake multi-views into textures
        print("5. Baking multi-views into textures...")
        cleanup_memory()  # Clean memory before baking
        bake_results = self.bake_multiviews.process(
            pipeline=pipeline,
            camera_config=camera_config,
            albedo=albedo,
            mr=mr,
        )

        baked_pipeline = bake_results[0]
        baked_albedo = bake_results[1]
        albedo_mask = bake_results[2]
        baked_mr = bake_results[3] if len(bake_results) > 3 else None
        mr_mask = bake_results[4] if len(bake_results) > 4 else None

        print("   Multi-view baking completed")
        cleanup_memory()  # Clean up after baking

        # Step 6: Load upscale model (if upscaling is enabled)
        upscale_model = None
        if upscale_albedo or upscale_mr:
            print("6. Loading upscale model...")
            upscale_model = self.upscale_model_loader.loadmodel(upscale_model_name)[0]
            print(f"   Upscale model loaded: {upscale_model_name}")

        # Step 7: Upscale albedo texture (if enabled)
        final_albedo = baked_albedo
        if upscale_albedo and upscale_model is not None:
            print("7. Upscaling albedo texture...")
            upscaled_albedo = self.upscale_image.upscale(
                upscale_model=upscale_model, image=baked_albedo
            )[0]
            final_albedo = upscaled_albedo
            print("   Albedo texture upscaled")
        else:
            print("7. Skipping albedo upscaling")

        # Step 8: Upscale MR texture (if enabled and available)
        final_mr = baked_mr
        if upscale_mr and upscale_model is not None and baked_mr is not None:
            print("8. Upscaling metallic-roughness texture...")
            upscaled_mr = self.upscale_image.upscale(
                upscale_model=upscale_model, image=baked_mr
            )[0]
            final_mr = upscaled_mr
            print("   Metallic-roughness texture upscaled")
        else:
            print("8. Skipping MR upscaling")

        # Step 9: Perform inpainting
        print("9. Performing texture inpainting...")
        cleanup_memory()  # Clean memory before inpainting
        inpaint_results = self.inpaint.process(
            pipeline=baked_pipeline,
            albedo=final_albedo,
            albedo_mask=albedo_mask,
            mr=final_mr if final_mr is not None else final_albedo,
            mr_mask=mr_mask if mr_mask is not None else albedo_mask,
            output_mesh_name=output_mesh_name,
        )

        inpainted_albedo = inpaint_results[0]
        inpainted_mr = inpaint_results[1]
        textured_trimesh = inpaint_results[2]  # The actual textured mesh!
        output_glb_path = inpaint_results[3]  # The correct path

        print("   Texture inpainting completed")
        cleanup_memory()  # Clean up after inpainting

        # Step 10: Export final textured mesh
        print("10. Exporting final textured mesh...")
        cleanup_memory()  # Clean memory before export
        final_output_path = self.export_mesh.process(
            trimesh=textured_trimesh,  # Use the textured mesh from inpaint!
            filename_prefix=f"3D/{output_mesh_name}_final",
            file_format=file_format,
            save_file=save_file,
        )[0]
        print(f"    Final textured mesh exported to: {final_output_path}")
        cleanup_memory()  # Final cleanup

        # Return results
        results = {
            "uv_wrapped_mesh": uv_wrapped_mesh,
            "textured_mesh": textured_trimesh,  # The actual textured mesh
            "camera_config": camera_config,
            "multiview_images": albedo,  # Use albedo images as multiview images
            "baked_albedo": baked_albedo,
            "baked_mr": baked_mr,
            "albedo_mask": albedo_mask,
            "mr_mask": mr_mask,
            "final_albedo": final_albedo,
            "final_mr": final_mr,
            "inpainted_albedo": inpainted_albedo,
            "inpainted_mr": inpainted_mr,
            "output_glb_path": output_glb_path,
            "final_output_path": final_output_path,
            "output_mesh_name": output_mesh_name,
        }

        print("Texture generation workflow completed successfully!")
        print(f"Generated {len(camera_azimuths.split(','))} views")
        print(f"Texture resolution: {texture_size}x{texture_size}")
        print(f"Final mesh: {final_output_path}")

        return results


class CompleteHunyuan3DWorkflow:
    """Combined workflow that generates mesh and then applies textures"""

    def __init__(self):
        self.mesh_workflow = ManualHunyuan3DWorkflow()
        self.texture_workflow = ManualHunyuan3DTextureWorkflow()

    def run_complete_workflow(
        self,
        vae_model_name,
        diffusion_model_name,
        input_image_path,
        output_mesh_name="complete_mesh",
        # Mesh generation parameters
        mesh_params=None,
        # Texture generation parameters
        texture_params=None,
        # Background removal parameters
        remove_background=False,
        bg_model_path=None,
        bg_threshold=0.5,
        bg_use_jit=False,
    ):

        print("Starting Complete Hunyuan 3D 2.1 Workflow (Mesh + Texture)...")

        # Default parameters optimized for 16GB VRAM
        if mesh_params is None:
            mesh_params = {
                "steps": 30,  # Reduced for 16GB VRAM
                "guidance_scale": 4.0,  # Reduced for stability
                "seed": 42,
                "max_facenum": 20000,  # Reduced for memory efficiency
                "octree_resolution": 256,  # Reduced for 16GB VRAM
                "num_chunks": 4000,  # Reduced for memory efficiency
                "enable_flash_vdm": True,  # Memory efficient
                "force_offload": True,  # Enable offloading
            }

        if texture_params is None:
            texture_params = {
                "view_size": 512,  # Reduced from 1024 for 16GB VRAM
                "steps": 20,  # Reduced from 35
                "guidance_scale": 4.0,  # Reduced from 5.6
                "texture_size": 2048,  # Reduced from 4096
                "upscale_albedo": True,  # Enable upscaling for quality
                "upscale_mr": True,
                "camera_azimuths": "0, 180, 90, 270, 45, 315",  # 6 views for better coverage
                "camera_elevations": "0, 0, 0, 0, 30, 30",
                "view_weights": "1.0, 1.0, 1.0, 1.0, 0.8, 0.8",
                "ortho_scale": 1.10,
                "normal_texture": True,
                "unwrap_mesh": True,
                "save_after_generate": False,
                "correct_after_generate": "randomize",
                "seed": 200434251488993,
                # InPaint parameters
                "vertex_inpaint": True,
                "method": "NS",
            }

        # Step 1: Generate mesh
        print("\n" + "=" * 60)
        print("PHASE 1: MESH GENERATION")
        print("=" * 60)

        mesh_results = self.mesh_workflow.run_workflow(
            vae_model_name=vae_model_name,
            diffusion_model_name=diffusion_model_name,
            input_image_path=input_image_path,
            output_mesh_name=f"{output_mesh_name}_base",
            remove_background=remove_background,
            bg_model_path=bg_model_path,
            bg_threshold=bg_threshold,
            bg_use_jit=bg_use_jit,
            **mesh_params,
        )

        # Step 2: Generate textures
        print("\n" + "=" * 60)
        print("PHASE 2: TEXTURE GENERATION")
        print("=" * 60)

        texture_results = self.texture_workflow.run_texture_workflow(
            input_trimesh=mesh_results["processed_mesh"],
            input_image_path=input_image_path,
            output_mesh_name=f"{output_mesh_name}_textured",
            remove_background=remove_background,
            bg_model_path=bg_model_path,
            bg_threshold=bg_threshold,
            bg_use_jit=bg_use_jit,
            **texture_params,
        )

        # Combine results
        complete_results = {
            "mesh_results": mesh_results,
            "texture_results": texture_results,
            "final_mesh_path": texture_results["final_output_path"],
            "base_mesh_path": mesh_results["output_path"],
        }

        # Print final summary
        print("\n" + "=" * 60)
        print("COMPLETE WORKFLOW SUMMARY")
        print("=" * 60)
        print(f"Input image: {input_image_path}")
        print(f"Base mesh: {mesh_results['output_path']}")
        print(f"Textured mesh: {texture_results['final_output_path']}")
        print(f"Original faces: {mesh_results['num_faces_before']:,}")
        print(f"Final faces: {mesh_results['num_faces_after']:,}")
        print(f"Generated views: {len(texture_results['multiview_images'])}")
        print("=" * 60)

        return complete_results


class EnhancedHunyuan3DWorkflow:
    """Enhanced workflow with mesh decimation and better memory management"""

    def __init__(self):
        self.mesh_workflow = ManualHunyuan3DWorkflow()
        self.texture_workflow = ManualHunyuan3DTextureWorkflow()
        self.mesh_decimator = Hy3D21SimpleMeshlibDecimate()

    def run_enhanced_workflow(
        self,
        vae_model_name,
        diffusion_model_name,
        input_image_path,
        output_mesh_name="enhanced_mesh",
        # Mesh generation parameters (optimized for 16GB VRAM)
        mesh_params=None,
        # Texture generation parameters (optimized for 16GB VRAM)
        texture_params=None,
        # Mesh decimation parameters
        enable_decimation=False,
        target_face_count=15000,  # Further reduce for 16GB VRAM
        preserve_boundary=True,
        boundary_weight=1.0,
        preserve_topology=True,
        # Background removal parameters
        remove_background=True,
        bg_model_path=None,
        bg_params=None,
    ):

        print("Starting Enhanced Hunyuan 3D 2.1 Workflow (Mesh + Texture + Decimation)...")

        # Handle background parameters
        if bg_params is None:
            bg_params = {
                "threshold": 0.5,
                "use_jit": False,
            }
        
        bg_threshold = bg_params.get("threshold", 0.5)
        bg_use_jit = bg_params.get("use_jit", False)

        # Default parameters optimized for 16GB VRAM
        if mesh_params is None:
            mesh_params = {
                "steps": 25,  # Further reduced
                "guidance_scale": 3.5,  # Further reduced
                "seed": 42,
                "max_facenum": 400000,  # Reduced
                "octree_resolution": 224,  # Further reduced
                "num_chunks": 3000,  # Further reduced
                "enable_flash_vdm": True,
                "force_offload": True,
            }
            # 32+ GB VRAM
            # mesh_params = {
            #     "steps": 50,
            #     "guidance_scale": 5.0,
            #     "octree_resolution": 384,
            #     "num_chunks": 8000,
            #     "max_facenum": 500000,    # Higher face count
            # }

        if texture_params is None:
            texture_params = {
                "view_size": 512,
                "steps": 15,  # Further reduced
                "guidance_scale": 3.5,  # Further reduced
                "texture_size": 1024,  # Further reduced for 16GB VRAM
                "upscale_albedo": True,  # Disable upscaling to save memory
                "upscale_mr": True,  # Disable upscaling to save memory
                # Camera configuration options
                "camera_azimuths": "0, 180, 90, 270, 45, 315",  # 6 views for better coverage
                "camera_elevations": "0, 0, 0, 0, 30, 30",
                "view_weights": "1.0, 1.0, 1.0, 1.0, 0.8, 0.8",
                "ortho_scale": 1.10,
                "normal_texture": True,
                "unwrap_mesh": True,
                "save_after_generate": False,
                "correct_after_generate": "randomize",
                "seed": 200434251488993,
                # InPaint parameters
                "vertex_inpaint": True,
                "method": "NS",
            }

        # Step 1: Generate base mesh
        print("\n" + "=" * 60)
        print("PHASE 1: MESH GENERATION")
        print("=" * 60)

        mesh_results = self.mesh_workflow.run_workflow(
            vae_model_name=vae_model_name,
            diffusion_model_name=diffusion_model_name,
            input_image_path=input_image_path,
            output_mesh_name=f"{output_mesh_name}_base",
            remove_background=remove_background,
            bg_model_path=bg_model_path,
            bg_threshold=bg_threshold,
            bg_use_jit=bg_use_jit,
            **mesh_params,
        )

        processed_mesh = mesh_results["processed_mesh"]

        # Step 2: Optional mesh decimation for better memory efficiency
        if enable_decimation:
            print("\n" + "=" * 60)
            print("PHASE 1.5: MESH DECIMATION")
            print("=" * 60)

            # Use 80% of max_facenum from mesh_params if available, otherwise use target_face_count
            if mesh_params and "max_facenum" in mesh_params:
                dynamic_target = int(mesh_params["max_facenum"] * 0.8)
                print(f"   Using dynamic target: {dynamic_target} faces (80% of max_facenum: {mesh_params['max_facenum']})")
            else:
                dynamic_target = target_face_count
                print(f"   Using fixed target: {dynamic_target} faces")

            print(f"   Decimating mesh from {mesh_results['num_faces_after']} to ~{dynamic_target} faces...")

            decimated_mesh = self.mesh_decimator.decimate(
                trimesh=processed_mesh,
                subdivideParts=8,  # Use 8 CPU cores
                target_face_num=dynamic_target,
            )[0]

            final_mesh_for_texture = decimated_mesh
            final_face_count = decimated_mesh.faces.shape[0]
            print(f"   Mesh decimated to {final_face_count} faces")

        else:
            final_mesh_for_texture = processed_mesh
            final_face_count = mesh_results['num_faces_after']

        # Clear memory before texture generation
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Step 3: Generate textures
        print("\n" + "=" * 60)
        print("PHASE 2: TEXTURE GENERATION")
        print("=" * 60)

        texture_results = self.texture_workflow.run_texture_workflow(
            input_trimesh=final_mesh_for_texture,
            input_image_path=input_image_path,
            output_mesh_name=f"{output_mesh_name}_textured",
            remove_background=remove_background,
            bg_model_path=bg_model_path,
            bg_threshold=bg_threshold,
            bg_use_jit=bg_use_jit,
            **texture_params,
        )

        # Combine results
        enhanced_results = {
            "mesh_results": mesh_results,
            "texture_results": texture_results,
            "decimated_mesh": final_mesh_for_texture if enable_decimation else None,
            "final_face_count": final_face_count,
            "final_mesh_path": texture_results["final_output_path"],
            "base_mesh_path": mesh_results["output_path"],
        }

        # Print final summary
        print("\n" + "=" * 60)
        print("ENHANCED WORKFLOW SUMMARY")
        print("=" * 60)
        print(f"Input image: {input_image_path}")
        print(f"Base mesh: {mesh_results['output_path']}")
        if enable_decimation:
            print(f"Decimated mesh faces: {final_face_count}")
        print(f"Textured mesh: {texture_results['final_output_path']}")
        print(f"Original faces: {mesh_results['num_faces_before']:,}")
        print(f"Processed faces: {mesh_results['num_faces_after']:,}")
        if enable_decimation:
            print(f"Final faces: {final_face_count:,}")
        print(f"Generated views: {len(texture_results['multiview_images'])}")
        print(f"Texture size: {texture_params['texture_size']}x{texture_params['texture_size']}")
        print("=" * 60)

        return enhanced_results


def main():
    """Example usage of the complete workflow"""
    import argparse

    parser = argparse.ArgumentParser(description="Hunyuan 3D 2.1 Manual Workflow")
    parser.add_argument(
        "--workflow",
        choices=["mesh", "texture", "complete", "enhanced"],
        default="enhanced",
        help="Which workflow to run (enhanced is recommended for 16GB VRAM)",
    )
    parser.add_argument(
        "--input-image", type=str, default="assets/mune.png", help="Input image path"
    )
    parser.add_argument(
        "--output-name", type=str, default="generated_model", help="Output mesh name"
    )
    parser.add_argument(
        "--vae-model", type=str, default="model.fp16.ckpt", help="VAE model filename"
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default="model.fp16.ckpt",
        help="Diffusion model filename",
    )
    parser.add_argument(
        "--remove-background", action="store_true", help="Remove background from input image"
    )
    parser.add_argument(
        "--bg-model-path", type=str, default="models/RMBG/INSPYRENET/inspyrenet.safetensors", 
        help="Path to background removal model"
    )
    parser.add_argument(
        "--bg-threshold", type=float, default=0.5, 
        help="Background removal threshold (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--bg-use-jit", action="store_true", 
        help="Use PyTorch JIT for faster inference"
    )

    args = parser.parse_args()

    try:
        if args.workflow == "mesh":
            # Run mesh generation only
            workflow = ManualHunyuan3DWorkflow()
            config = {
                "vae_model_name": args.vae_model,
                "diffusion_model_name": args.diffusion_model,
                "input_image_path": args.input_image,
                "output_mesh_name": args.output_name,
                "steps": 25,
                "guidance_scale": 3.5,
                "seed": 42,
                "max_facenum": 20000,
                "octree_resolution": 224,
                "num_chunks": 3000,
                "enable_flash_vdm": True,
                "force_offload": True,
                "file_format": "glb",
                "save_file": True,
                **({"remove_background": args.remove_background} if args.remove_background else {}),
                **({"bg_model_path": args.bg_model_path} if args.bg_model_path != parser.get_default("bg_model_path") else {}),
                **({"bg_threshold": args.bg_threshold} if args.bg_threshold != parser.get_default("bg_threshold") else {}),
                **({"bg_use_jit": args.bg_use_jit} if args.bg_use_jit else {}),
            }
            results = workflow.run_workflow(**config)

            # Print summary
            print("\n" + "=" * 50)
            print("MESH GENERATION SUMMARY")
            print("=" * 50)
            print(f"Input image: {config['input_image_path']}")
            print(f"Output mesh: {results['output_path']}")
            print(f"Original faces: {results['num_faces_before']:,}")
            print(f"Final faces: {results['num_faces_after']:,}")
            print(
                f"Face reduction: {((results['num_faces_before'] - results['num_faces_after']) / results['num_faces_before'] * 100):.1f}%"
            )

        elif args.workflow == "texture":
            # Run texture generation only (requires existing mesh)
            print("Texture-only workflow requires an existing trimesh object.")
            print("Please use the complete or enhanced workflow instead.")

        elif args.workflow == "complete":
            # Run complete workflow
            workflow = CompleteHunyuan3DWorkflow()
            results = workflow.run_complete_workflow(
                vae_model_name=args.vae_model,
                diffusion_model_name=args.diffusion_model,
                input_image_path=args.input_image,
                output_mesh_name=args.output_name,
                **({"remove_background": args.remove_background} if args.remove_background else {}),
                **({"bg_model_path": args.bg_model_path} if args.bg_model_path != parser.get_default("bg_model_path") else {}),
                **({"bg_threshold": args.bg_threshold} if args.bg_threshold != parser.get_default("bg_threshold") else {}),
                **({"bg_use_jit": args.bg_use_jit} if args.bg_use_jit else {}),
            )
            print("\nComplete workflow finished!")
            print(f"Final textured mesh: {results['final_mesh_path']}")

        elif args.workflow == "enhanced":
            # Run enhanced workflow (recommended for 16GB VRAM)
            workflow = EnhancedHunyuan3DWorkflow()

            # Default texture parameters with camera configurations
            default_texture_params = {
                "view_size": 512,
                "steps": 15,
                "guidance_scale": 3.5,
                "texture_size": 1024,
                "upscale_albedo": False,
                "upscale_mr": False,
                "camera_azimuths": "0, 180, 90, 270, 45, 315",
                "camera_elevations": "0, 0, 0, 0, 30, 30",
                "view_weights": "1.0, 1.0, 1.0, 1.0, 0.8, 0.8",
                "ortho_scale": 1.10,
            }
            # 32+ GB VRAM
            # texture_params = {
            #     "view_size": 1024,
            #     "texture_size": 4096,     # Maximum texture resolution
            #     "steps": 35,             # Maximum quality
            #     "guidance_scale": 5.6,   # Maximum adherence
            #     "upscale_albedo": True,
            #     "upscale_mr": True,
            # }

            results = workflow.run_enhanced_workflow(
                vae_model_name=args.vae_model,
                diffusion_model_name=args.diffusion_model,
                input_image_path=args.input_image,
                output_mesh_name=args.output_name,
                texture_params=default_texture_params,  # Pass texture params with camera configs
                **({"remove_background": args.remove_background} if args.remove_background else {}),
                **({"bg_model_path": args.bg_model_path} if args.bg_model_path != parser.get_default("bg_model_path") else {}),
                bg_params={
                    "threshold": args.bg_threshold,
                    "use_jit": args.bg_use_jit,
                },
            )
            print("\nEnhanced workflow finished!")
            print(f"Final textured mesh: {results['final_mesh_path']}")
            if results['decimated_mesh'] is not None:
                print(f"Mesh was decimated to {results['final_face_count']:,} faces for optimal performance")

    except Exception as e:
        print(f"Error running workflow: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()


# Example usage of the optimized workflow for 16GB VRAM

# Uncomment the code below to run examples:

# # 1. Basic mesh generation (fastest, lowest memory)
# from manual_workflow import ManualHunyuan3DWorkflow
#
# mesh_workflow = ManualHunyuan3DWorkflow()
# results = mesh_workflow.run_workflow(
#     vae_model_name="model.fp16.ckpt",
#     diffusion_model_name="model.fp16.ckpt",
#     input_image_path="assets/mune.png",
#     output_mesh_name="my_mesh",
#     steps=25,  # Optimized for 16GB VRAM
#     guidance_scale=3.5,
#     octree_resolution=224,
#     num_chunks=3000,
#     enable_flash_vdm=True,
#     force_offload=True,
# )

# # 2. Complete workflow (mesh + texture)
# from manual_workflow import CompleteHunyuan3DWorkflow
#
# complete_workflow = CompleteHunyuan3DWorkflow()
# results = complete_workflow.run_complete_workflow(
#     vae_model_name="model.fp16.ckpt",
#     diffusion_model_name="model.fp16.ckpt",
#     input_image_path="assets/mune.png",
#     output_mesh_name="my_textured_mesh",
#     mesh_params={
#         "steps": 25,
#         "guidance_scale": 3.5,
#         "max_facenum": 20000,
#         "octree_resolution": 224,
#         "num_chunks": 3000,
#         "enable_flash_vdm": True,
#         "force_offload": True,
#     },
#     texture_params={
#         "view_size": 512,
#         "steps": 15,
#         "guidance_scale": 3.5,
#         "texture_size": 1024,
#         "upscale_albedo": False,  # Save memory
#         "upscale_mr": False,
#     }
# )

# # 3. Enhanced workflow (recommended - includes mesh decimation)
# from manual_workflow import EnhancedHunyuan3DWorkflow
#
# enhanced_workflow = EnhancedHunyuan3DWorkflow()
# results = enhanced_workflow.run_enhanced_workflow(
#     vae_model_name="model.fp16.ckpt",
#     diffusion_model_name="model.fp16.ckpt",
#     input_image_path="assets/mune.png",
#     output_mesh_name="my_optimized_mesh",
#     # Mesh decimation is enabled by default
#     target_face_count=15000,  # Optimal for 16GB VRAM
# )
