"""
RunPod Serverless Handler for Hunyuan 3D 2.1 Workflow
=====================================================

This module implements an async RunPod serverless handler for the Hunyuan 3D 2.1 workflow.
It supports mesh generation, texture generation, and enhanced workflows.

SUPPORTED WORKFLOWS:
1. "mesh" - Mesh generation only (fastest)
2. "texture" - Texture generation for existing meshes (requires base64 mesh data)
3. "enhanced" - Complete pipeline with mesh + texture + decimation (recommended)

INPUT FORMAT:
{
    "workflow": "enhanced",  # Required: "mesh", "texture", or "enhanced"
    "input_image": "base64_encoded_image_data",  # Required: Base64 encoded image
    "output_name": "my_mesh",  # Optional: Output filename prefix
    "vae_model": "model.fp16.ckpt",  # Optional: VAE model filename
    "diffusion_model": "model.fp16.ckpt",  # Optional: Diffusion model filename

    # Background removal options
    "remove_background": true,  # Optional: Enable background removal
    "bg_threshold": 0.5,  # Optional: Background removal threshold (0.0-1.0)
    "bg_use_jit": false,  # Optional: Use PyTorch JIT for faster inference

    # Mesh generation parameters (for mesh/enhanced workflows)
    "mesh_params": {
        "steps": 25,
        "guidance_scale": 3.5,
        "seed": 42,
        "max_facenum": 20000,
        "octree_resolution": 224,
        "num_chunks": 3000,
        "enable_flash_vdm": true,
        "force_offload": true
    },

    # Texture generation parameters (for texture/enhanced workflows)
    "texture_params": {
        "view_size": 512,
        "steps": 15,
        "guidance_scale": 3.5,
        "texture_size": 1024,
        "upscale_albedo": false,
        "upscale_mr": false,
        "camera_azimuths": "0, 180, 90, 270, 45, 315",
        "camera_elevations": "0, 0, 0, 0, 30, 30",
        "view_weights": "1.0, 1.0, 1.0, 1.0, 0.8, 0.8",
        "ortho_scale": 1.10
    },

    # Decimation parameters (for enhanced workflow)
    "decimation_params": {
        "enable_decimation": false,
        "target_face_count": 15000
    }
}

OUTPUT FORMAT:
{
    "status": "success",
    "workflow_type": "enhanced",
    "output_files": [
        {
            "filename": "enhanced_mesh_base.glb",
            "download_url": "https://...",
            "file_type": "base_mesh"
        },
        {
            "filename": "enhanced_mesh_textured_final.glb",
            "download_url": "https://...",
            "file_type": "textured_mesh"
        }
    ],
    "mesh_stats": {
        "original_faces": 45000,
        "processed_faces": 20000,
        "final_faces": 15000
    },
    "processing_time": 120.5,
    "memory_optimized": true
}

ENVIRONMENT REQUIREMENTS:
- All dependencies from requirements.txt must be installed
- Models should be placed in the appropriate directories
- RunPod SDK: pip install runpod
"""

import asyncio
import base64
import io
import json
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Union

import runpod
from PIL import Image

# Import the workflow classes
from manual_workflow import (
    ManualHunyuan3DWorkflow,
    ManualHunyuan3DTextureWorkflow,
    EnhancedHunyuan3DWorkflow,
    cleanup_memory,
)


class RunPodHunyuan3DHandler:
    """Async RunPod handler for Hunyuan 3D workflows"""

    def __init__(self):
        """Initialize the handler and load models"""
        print("🚀 Initializing RunPod Hunyuan 3D Handler...")

        # Initialize workflow instances (models will be loaded on-demand)
        self.mesh_workflow: Union[None, ManualHunyuan3DWorkflow] = None
        self.texture_workflow = None
        self.enhanced_workflow: Union[None, EnhancedHunyuan3DWorkflow] = None

        # Default model paths
        self.default_vae_model = "model.fp16.ckpt"
        self.default_diffusion_model = "model.fp16.ckpt"

        # Ensure output directory exists
        self.output_dir = Path("output/3D")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("✅ RunPod Hunyuan 3D Handler initialized successfully")

    def _initialize_workflow(self, workflow_type: str):
        """Lazy initialization of workflow instances"""
        if workflow_type == "mesh" and self.mesh_workflow is None:
            print("🔄 Initializing mesh workflow...")
            self.mesh_workflow = ManualHunyuan3DWorkflow()

        elif workflow_type == "texture" and self.texture_workflow is None:
            print("🔄 Initializing texture workflow...")
            self.texture_workflow = ManualHunyuan3DTextureWorkflow()

        elif workflow_type == "enhanced" and self.enhanced_workflow is None:
            print("🔄 Initializing enhanced workflow...")
            self.enhanced_workflow = EnhancedHunyuan3DWorkflow()

    def _decode_base64_image(self, base64_data: str) -> Image.Image:
        """Decode base64 image data to PIL Image"""
        try:
            # Remove data URL prefix if present
            if base64_data.startswith("data:image"):
                base64_data = base64_data.split(",")[1]

            # Decode base64 data
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            print(f"📷 Image decoded: {image.size} pixels")
            return image

        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")

    def _save_image_temporarily(
        self, image: Image.Image, name: str = "temp_input"
    ) -> str:
        """Save PIL Image temporarily for processing"""
        temp_path = f"/tmp/{name}.png"
        image.save(temp_path)
        return temp_path

    def _validate_input(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize job input"""
        # Required fields
        if "workflow" not in job_input:
            raise ValueError("Missing required field: 'workflow'")

        if "input_image" not in job_input:
            raise ValueError("Missing required field: 'input_image'")

        workflow_type = job_input["workflow"]
        if workflow_type not in ["mesh", "texture", "enhanced"]:
            raise ValueError(
                f"Invalid workflow type: {workflow_type}. Must be one of: mesh, texture, enhanced"
            )

        # Normalize input
        normalized = {
            "workflow": workflow_type,
            "input_image": job_input["input_image"],
            "output_name": job_input.get("output_name", "generated_model"),
            "vae_model": job_input.get("vae_model", self.default_vae_model),
            "diffusion_model": job_input.get(
                "diffusion_model", self.default_diffusion_model
            ),
            "remove_background": job_input.get("remove_background", True),
            "bg_threshold": job_input.get("bg_threshold", 0.5),
            "bg_use_jit": job_input.get("bg_use_jit", False),
        }

        # Add workflow-specific parameters
        if workflow_type in ["mesh", "enhanced"]:
            normalized["mesh_params"] = job_input.get("mesh_params", {})

        if workflow_type in ["texture", "enhanced"]:
            normalized["texture_params"] = job_input.get("texture_params", {})

        if workflow_type == "enhanced":
            normalized["decimation_params"] = job_input.get("decimation_params", {})

        return normalized

    def _get_default_mesh_params(self) -> Dict[str, Any]:
        """Get default mesh generation parameters optimized for serverless"""
        return {
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
        }

    def _get_default_texture_params(self) -> Dict[str, Any]:
        """Get default texture generation parameters optimized for serverless"""
        return {
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

    def _get_default_decimation_params(self) -> Dict[str, Any]:
        """Get default decimation parameters"""
        return {
            "enable_decimation": False,
            "target_face_count": 15000,
        }

    async def _run_mesh_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run mesh generation workflow"""
        print("🔵 Starting mesh generation workflow...")

        # Initialize workflow
        self._initialize_workflow("mesh")

        # Decode and save input image
        image = self._decode_base64_image(params["input_image"])
        temp_image_path = self._save_image_temporarily(
            image, f"{params['output_name']}_input"
        )

        # Prepare mesh parameters
        mesh_params = self._get_default_mesh_params()
        mesh_params.update(params.get("mesh_params", {}))

        if self.mesh_workflow is None:
            raise RuntimeError("Mesh workflow not initialized")

        # Run workflow
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.mesh_workflow.run_workflow(
                vae_model_name=params["vae_model"],
                diffusion_model_name=params["diffusion_model"],
                input_image_path=temp_image_path,
                output_mesh_name=params["output_name"],
                remove_background=params["remove_background"],
                bg_threshold=params["bg_threshold"],
                bg_use_jit=params["bg_use_jit"],
                **mesh_params,
            ),
        )

        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        return {
            "workflow_type": "mesh",
            "output_files": [
                {
                    "filename": os.path.basename(results["output_path"]),
                    "full_path": results["output_path"],
                    "file_type": "mesh",
                }
            ],
            "mesh_stats": {
                "original_faces": results["num_faces_before"],
                "processed_faces": results["num_faces_after"],
                "final_faces": results["num_faces_after"],
            },
            "vram_optimized": results.get("vram_optimized", True),
        }

    async def _run_enhanced_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run enhanced workflow (mesh + texture + decimation)"""
        print("🟢 Starting enhanced workflow...")

        # Initialize workflow
        self._initialize_workflow("enhanced")

        # Decode and save input image
        image = self._decode_base64_image(params["input_image"])
        temp_image_path = self._save_image_temporarily(
            image, f"{params['output_name']}_input"
        )

        # Prepare parameters
        mesh_params = self._get_default_mesh_params()
        mesh_params.update(params.get("mesh_params", {}))

        texture_params = self._get_default_texture_params()
        texture_params.update(params.get("texture_params", {}))

        decimation_params = self._get_default_decimation_params()
        decimation_params.update(params.get("decimation_params", {}))

        if self.enhanced_workflow is None:
            raise RuntimeError("Enhanced workflow not initialized")

        # Run enhanced workflow
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.enhanced_workflow.run_enhanced_workflow(
                vae_model_name=params["vae_model"],
                diffusion_model_name=params["diffusion_model"],
                input_image_path=temp_image_path,
                output_mesh_name=params["output_name"],
                mesh_params=mesh_params,
                texture_params=texture_params,
                remove_background=params["remove_background"],
                bg_params={
                    "threshold": params["bg_threshold"],
                    "use_jit": params["bg_use_jit"],
                },
                **decimation_params,
            ),
        )

        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Prepare output files list
        output_files = []

        # Base mesh
        if "base_mesh_path" in results:
            output_files.append(
                {
                    "filename": os.path.basename(results["base_mesh_path"]),
                    "full_path": results["base_mesh_path"],
                    "file_type": "base_mesh",
                }
            )

        # Final textured mesh
        if "final_mesh_path" in results:
            output_files.append(
                {
                    "filename": os.path.basename(results["final_mesh_path"]),
                    "full_path": results["final_mesh_path"],
                    "file_type": "textured_mesh",
                }
            )

        return {
            "workflow_type": "enhanced",
            "output_files": output_files,
            "mesh_stats": {
                "original_faces": results["mesh_results"]["num_faces_before"],
                "processed_faces": results["mesh_results"]["num_faces_after"],
                "final_faces": results["final_face_count"],
            },
            "texture_info": {
                "generated_views": len(results["texture_results"]["multiview_images"]),
                "texture_size": texture_params["texture_size"],
            },
        }

    async def process_job(
        self, job_input: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a job asynchronously with progress updates"""
        start_time = time.time()

        try:
            # Validate input
            params = self._validate_input(job_input)
            workflow_type = params["workflow"]

            results = {}

            yield {
                "status": "processing",
                "progress": 0,
                "message": f"Starting {workflow_type} workflow...",
                "workflow_type": workflow_type,
            }

            # Memory cleanup before starting
            cleanup_memory()

            # Run appropriate workflow
            if workflow_type == "mesh":
                yield {
                    "status": "processing",
                    "progress": 10,
                    "message": "Generating 3D mesh from image...",
                }
                results = await self._run_mesh_workflow(params)

            elif workflow_type == "enhanced":
                yield {
                    "status": "processing",
                    "progress": 10,
                    "message": "Starting enhanced workflow (mesh + texture)...",
                }
                results = await self._run_enhanced_workflow(params)

            elif workflow_type == "texture":
                yield {
                    "status": "error",
                    "message": "Texture-only workflow not yet implemented for serverless. Use 'enhanced' workflow instead.",
                }
                return

            # Final cleanup
            cleanup_memory()

            processing_time = time.time() - start_time

            # Final result
            final_result = {
                "status": "success",
                "processing_time": round(processing_time, 2),
                **results,
            }

            yield {
                "status": "processing",
                "progress": 100,
                "message": "Processing complete!",
            }

            yield final_result

        except Exception as e:
            error_message = str(e)
            error_traceback = traceback.format_exc()

            print(f"❌ Error processing job: {error_message}")
            print(f"Traceback: {error_traceback}")

            yield {
                "status": "error",
                "error": error_message,
                "traceback": error_traceback,
                "processing_time": round(time.time() - start_time, 2),
            }


# Global handler instance
handler_instance = RunPodHunyuan3DHandler()


async def runpod_handler(job: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Main RunPod async handler function

    Args:
        job: RunPod job dictionary containing 'id' and 'input' fields

    Yields:
        Progress updates and final results
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})

    print(f"🎯 Processing RunPod job {job_id}")
    print(f"📝 Input: {json.dumps(job_input, indent=2)}")

    # Process the job and yield results
    async for result in handler_instance.process_job(job_input):
        yield result


# Test function for local development
def test_handler_locally():
    """Test the handler locally with sample input"""

    test_input = {
        "id": "test_job_123",
        "input": {
            "workflow": "mesh",
            "input_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # 1x1 red pixel
            "output_name": "test_mesh",
            "remove_background": False,
            "mesh_params": {"steps": 5, "max_facenum": 1000},  # Reduced for testing
        },
    }

    async def run_test():
        print("🧪 Running local test...")
        async for result in runpod_handler(test_input):
            print(f"📤 Result: {json.dumps(result, indent=2)}")

    asyncio.run(run_test())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run local test
        test_handler_locally()
    else:
        # Start RunPod serverless
        print("🚀 Starting RunPod Serverless Handler for Hunyuan 3D 2.1")
        runpod.serverless.start(
            {
                "handler": runpod_handler,
                "return_aggregate_stream": True,  # Make results available via /run endpoint
            }
        )
