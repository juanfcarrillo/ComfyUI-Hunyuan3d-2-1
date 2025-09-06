"""
RunPod Serverless Handler for Hunyuan 3D 2.1 Workflow
=====================================================

This module implements a synchronous RunPod serverless handler for the Hunyuan 3D 2.1 workflow.
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

    # Upload configuration
    "upload_to_r2": true,  # Optional: Upload results to R2 storage
    "keep_local_files": false,  # Optional: Keep local files after upload

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

import base64
import io
import json
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Union

import runpod
from PIL import Image

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print(
        "⚠️ python-dotenv not installed. Environment variables will only be loaded from system environment."
    )
    pass

# Import the workflow classes
from manual_workflow import (
    ManualHunyuan3DWorkflow,
    ManualHunyuan3DTextureWorkflow,
    EnhancedHunyuan3DWorkflow,
    cleanup_memory,
)

# Import R2 uploader
from r2_uploader import upload_3d_model, create_model_uploader


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

        # Initialize R2 uploader if configured
        self.r2_uploader = None
        self._check_r2_configuration()

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

    def _check_r2_configuration(self):
        """Check if R2 is properly configured"""
        required_r2_vars = [
            "R2_ACCOUNT_ID",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET_NAME",
        ]
        r2_configured = all(os.getenv(var) for var in required_r2_vars)

        if r2_configured:
            try:
                self.r2_uploader = create_model_uploader()
                print("✅ R2 storage configured and ready")
            except Exception as e:
                print(f"⚠️ R2 configuration found but connection failed: {e}")
                self.r2_uploader = None
        else:
            print("ℹ️ R2 storage not configured (missing environment variables)")

    def _decode_base64_image(self, base64_data: str) -> Image.Image:
        """Decode base64 image data to PIL Image, or load from file path"""
        try:
            # Check if it's a file path
            if not base64_data.startswith("data:image") and not base64_data.startswith(
                "iVBOR"
            ):
                # Assume it's a file path
                if os.path.exists(base64_data):
                    print(f"📁 Loading image from file: {base64_data}")
                    image = Image.open(base64_data).convert("RGB")
                    print(f"📷 Image loaded: {image.size} pixels")
                    return image
                else:
                    raise ValueError(f"File not found: {base64_data}")

            # Handle base64 data
            # Remove data URL prefix if present
            if base64_data.startswith("data:image"):
                base64_data = base64_data.split(",")[1]

            # Decode base64 data
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            print(f"📷 Image decoded from base64: {image.size} pixels")
            return image

        except Exception as e:
            raise ValueError(f"Failed to decode/load image: {str(e)}")

    def _save_image_temporarily(
        self, image: Image.Image, name: str = "temp_input"
    ) -> str:
        """Save PIL Image temporarily for processing"""
        # Use Windows-compatible temp directory
        import tempfile

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{name}.png")

        # Ensure temp directory exists
        os.makedirs(temp_dir, exist_ok=True)

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
            "upload_to_r2": job_input.get("upload_to_r2", True),
            "keep_local_files": job_input.get("keep_local_files", False),
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

    def _upload_files_to_r2(
        self, output_files: list, workflow_type: str, output_name: str
    ) -> list:
        """Upload output files to R2 and update file info with download URLs"""
        if not self.r2_uploader:
            print("ℹ️ R2 upload skipped - not configured")
            return output_files

        print("☁️ Uploading files to R2 storage...")

        uploaded_files = []

        for file_info in output_files:
            local_path = file_info["full_path"]
            filename = file_info["filename"]
            file_type = file_info["file_type"]

            try:
                # Create metadata for the upload
                metadata = {
                    "workflow_type": workflow_type,
                    "output_name": output_name,
                    "file_type": file_type,
                    "original_filename": filename,
                }

                # Upload file
                download_url = upload_3d_model(
                    local_file_path=local_path,
                    model_name=output_name,
                    workflow_type=workflow_type,
                    additional_metadata=metadata,
                )

                # Update file info with download URL
                updated_file_info = file_info.copy()
                updated_file_info["download_url"] = download_url
                updated_file_info["uploaded_to_r2"] = True

                uploaded_files.append(updated_file_info)
                print(f"✅ Uploaded {filename} to R2")

            except Exception as e:
                print(f"❌ Failed to upload {filename} to R2: {e}")
                # Keep original file info but mark upload as failed
                failed_file_info = file_info.copy()
                failed_file_info["upload_error"] = str(e)
                failed_file_info["uploaded_to_r2"] = False
                uploaded_files.append(failed_file_info)

        return uploaded_files

    def _cleanup_local_files(self, output_files: list):
        """Clean up local files after successful upload"""
        print("🧹 Cleaning up local files...")

        for file_info in output_files:
            if file_info.get("uploaded_to_r2", False):
                local_path = file_info["full_path"]
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                        print(f"🗑️ Removed local file: {os.path.basename(local_path)}")
                except Exception as e:
                    print(f"⚠️ Failed to remove local file {local_path}: {e}")

    def _run_mesh_workflow_sync(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run mesh generation workflow synchronously"""
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

        # Run workflow synchronously
        results = self.mesh_workflow.run_workflow(
            vae_model_name=params["vae_model"],
            diffusion_model_name=params["diffusion_model"],
            input_image_path=temp_image_path,
            output_mesh_name=params["output_name"],
            remove_background=params["remove_background"],
            bg_threshold=params["bg_threshold"],
            bg_use_jit=params["bg_use_jit"],
            **mesh_params,
        )

        # Clean up temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        output_files = [
            {
                "filename": os.path.basename(results["output_path"]),
                "full_path": results["output_path"],
                "file_type": "mesh",
            }
        ]

        # Upload to R2 if requested
        if params["upload_to_r2"]:
            output_files = self._upload_files_to_r2(
                output_files, "mesh", params["output_name"]
            )

            # Clean up local files if requested
            if not params["keep_local_files"]:
                self._cleanup_local_files(output_files)

        return {
            "workflow_type": "mesh",
            "output_files": output_files,
            "mesh_stats": {
                "original_faces": results["num_faces_before"],
                "processed_faces": results["num_faces_after"],
                "final_faces": results["num_faces_after"],
            },
            "vram_optimized": results.get("vram_optimized", True),
        }

    def _run_enhanced_workflow_sync(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run enhanced workflow synchronously"""
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

        # Run enhanced workflow synchronously
        results = self.enhanced_workflow.run_enhanced_workflow(
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

        # Upload to R2 if requested
        if params["upload_to_r2"]:
            output_files = self._upload_files_to_r2(
                output_files, "enhanced", params["output_name"]
            )

            # Clean up local files if requested
            if not params["keep_local_files"]:
                self._cleanup_local_files(output_files)

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

    def process_job_sync(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process a job synchronously"""
        start_time = time.time()

        try:
            # Validate input
            params = self._validate_input(job_input)
            workflow_type = params["workflow"]

            results = {}

            print(f"Starting {workflow_type} workflow...")

            # Memory cleanup before starting
            cleanup_memory()

            # Run appropriate workflow
            if workflow_type == "mesh":
                print("Generating 3D mesh from image...")
                results = self._run_mesh_workflow_sync(params)

            elif workflow_type == "enhanced":
                print("Starting enhanced workflow (mesh + texture)...")
                results = self._run_enhanced_workflow_sync(params)

            elif workflow_type == "texture":
                return {
                    "status": "error",
                    "message": "Texture-only workflow not yet implemented for serverless. Use 'enhanced' workflow instead.",
                }

            # Final cleanup
            cleanup_memory()

            processing_time = time.time() - start_time

            # Final result
            final_result = {
                "status": "success",
                "processing_time": round(processing_time, 2),
                "r2_configured": self.r2_uploader is not None,
                **results,
            }

            print("Processing complete!")
            return final_result

        except Exception as e:
            error_message = str(e)
            error_traceback = traceback.format_exc()

            print(f"❌ Error processing job: {error_message}")
            print(f"Traceback: {error_traceback}")

            return {
                "status": "error",
                "error": error_message,
                "traceback": error_traceback,
                "processing_time": round(time.time() - start_time, 2),
            }


# Global handler instance
handler_instance = RunPodHunyuan3DHandler()


def runpod_handler_sync(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod synchronous handler function

    Args:
        job: RunPod job dictionary containing 'id' and 'input' fields

    Returns:
        Final result dictionary
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})

    print(f"🎯 Processing RunPod job {job_id}")
    print(f"📝 Input: {json.dumps(job_input, indent=2)}")

    # Process the job synchronously
    return handler_instance.process_job_sync(job_input)


# ASYNC VERSION (commented out for now)
# async def runpod_handler(job: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
#     """
#     Main RunPod async handler function
#
#     Args:
#         job: RunPod job dictionary containing 'id' and 'input' fields
#
#     Yields:
#         Progress updates and final results
#     """
#     job_id = job.get("id", "unknown")
#     job_input = job.get("input", {})
#
#     print(f"🎯 Processing RunPod job {job_id}")
#     print(f"📝 Input: {json.dumps(job_input, indent=2)}")
#
#     # Process the job and yield results
#     async for result in handler_instance.process_job(job_input):
#         yield result


# Test function for local development
def test_handler_locally():
    """Test the handler locally with sample input from test_input.json"""
    import json

    print("🧪 Running local test with test_input.json...")

    try:
        with open("test_input.json", "r") as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load test_input.json: {e}")
        return

    result = runpod_handler_sync(test_data)
    print(f"📤 Result: {json.dumps(result, indent=2)}")


# Test function with hardcoded minimal data (for quick validation)
def test_handler_locally_minimal():
    """Test the handler locally with minimal hardcoded input"""

    test_input = {
        "id": "test_job_minimal",
        "input": {
            "workflow": "mesh",
            "input_image": "assets/mune.png",  # Use file path
            "output_name": "test_mesh_minimal",
            "remove_background": False,
            "mesh_params": {"steps": 5, "max_facenum": 1000},  # Reduced for testing
        },
    }

    print("🧪 Running minimal local test...")
    result = runpod_handler_sync(test_input)
    print(f"📤 Result: {json.dumps(result, indent=2)}")


# Test function with validation only (no actual processing)
def test_handler_validation():
    """Test the handler validation without running the actual workflow"""

    test_input = {
        "id": "test_job_validation",
        "input": {
            "workflow": "mesh",
            "input_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "output_name": "test_validation",
            "remove_background": False,
        },
    }

    print("🧪 Running validation test...")
    handler = RunPodHunyuan3DHandler()

    # Test input validation
    try:
        params = handler._validate_input(test_input["input"])
        print("✅ Input validation passed")
        print(f"📝 Validated params: {json.dumps(params, indent=2)}")

        # Test image decoding
        image = handler._decode_base64_image(params["input_image"])
        print(f"✅ Image decoding passed: {image.size} pixels")

        print("🎉 All validation tests passed!")

    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return False

    return True


# Test function for image decoding only
def test_image_decoding():
    """Test image decoding and saving functionality"""
    import json

    print("🧪 Testing image decoding functionality...")

    # Load test input
    try:
        with open("test_input.json", "r") as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load test_input.json: {e}")
        return False

    base64_image = test_data["input"]["input_image"]
    print(f"📝 Base64 image length: {len(base64_image)} characters")

    # Initialize handler
    handler = RunPodHunyuan3DHandler()

    try:
        # Test image decoding
        print("🔄 Decoding base64 image...")
        image = handler._decode_base64_image(base64_image)
        print(f"✅ Image decoded successfully: {image.size} pixels, mode: {image.mode}")

        # Test image saving
        print("💾 Testing image saving...")
        temp_path = handler._save_image_temporarily(image, "test_decoding")
        print(f"✅ Image saved to: {temp_path}")

        # Verify saved image
        if os.path.exists(temp_path):
            print("✅ Temporary file exists")
            file_size = os.path.getsize(temp_path)
            print(f"📊 File size: {file_size} bytes")

            # Try to reload the saved image
            try:
                saved_image = Image.open(temp_path)
                print(
                    f"✅ Saved image reloaded: {saved_image.size} pixels, mode: {saved_image.mode}"
                )

                # Clean up
                os.remove(temp_path)
                print("🧹 Cleaned up temporary file")

                return True

            except Exception as e:
                print(f"❌ Failed to reload saved image: {e}")
                return False
        else:
            print("❌ Temporary file was not created")
            return False

    except Exception as e:
        print(f"❌ Image decoding test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


# Test function with file path input (alternative to base64)
def test_file_input():
    """Test with file path input instead of base64"""
    print("🧪 Testing file path input...")

    # Use the assets/mune.png file
    image_path = "assets/mune.png"

    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False

    try:
        # Load image directly
        image = Image.open(image_path).convert("RGB")
        print(f"✅ Image loaded: {image.size} pixels, mode: {image.mode}")

        # Test saving
        handler = RunPodHunyuan3DHandler()
        temp_path = handler._save_image_temporarily(image, "test_file")
        print(f"✅ Image saved to: {temp_path}")

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("🧹 Cleaned up temporary file")

        return True

    except Exception as e:
        print(f"❌ File input test failed: {e}")
        return False


# Test function for R2 integration
def test_r2_integration():
    """Test R2 integration with the handler"""
    print("🧪 Testing R2 integration...")

    # Test input with R2 upload enabled
    test_input = {
        "workflow": "mesh",
        "input_image": "assets/mune.png",  # Use file path
        "output_name": "test_r2_upload",
        "upload_to_r2": True,
        "keep_local_files": False,
        "remove_background": False,
        "mesh_params": {"steps": 5, "max_facenum": 1000},  # Reduced for testing
    }

    handler = RunPodHunyuan3DHandler()
    result = handler.process_job_sync(test_input)

    print(f"📤 Test result: {json.dumps(result, indent=2)}")

    # Check if files were uploaded
    if result.get("status") == "success":
        for file_info in result.get("output_files", []):
            if file_info.get("uploaded_to_r2"):
                print(f"✅ File uploaded to R2: {file_info['download_url']}")
            else:
                print(
                    f"❌ File not uploaded: {file_info.get('upload_error', 'Unknown error')}"
                )


# ASYNC TEST VERSION (commented out)
# def test_handler_locally_async():
#     """Test the handler locally with sample input (async version)"""
#
#     test_input = {
#         "id": "test_job_123",
#         "input": {
#             "workflow": "mesh",
#             "input_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # 1x1 red pixel
#             "output_name": "test_mesh",
#             "remove_background": False,
#             "mesh_params": {"steps": 5, "max_facenum": 1000},  # Reduced for testing
#         },
#     }
#
#     async def run_test():
#         print("🧪 Running local test...")
#         async for result in runpod_handler(test_input):
#             print(f"📤 Result: {json.dumps(result, indent=2)}")
#
#     asyncio.run(run_test())

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Run local test with test_input.json
            test_handler_locally()
        elif sys.argv[1] == "--test-minimal":
            # Run minimal test with hardcoded data
            test_handler_locally_minimal()
        elif sys.argv[1] == "--validate":
            # Run validation test only
            test_handler_validation()
        elif sys.argv[1] == "--test-image":
            # Test image decoding only
            test_image_decoding()
        elif sys.argv[1] == "--test-file":
            # Test file input
            test_file_input()
        elif sys.argv[1] == "--test-r2":
            # Test R2 integration
            test_r2_integration()
        else:
            print(
                "Usage: python runpod_handler.py [--test|--test-minimal|--validate|--test-image|--test-file|--test-r2]"
            )
    else:
        # Start RunPod serverless with synchronous handler
        print("🚀 Starting RunPod Serverless Handler for Hunyuan 3D 2.1 (Synchronous)")
        runpod.serverless.start(
            {
                "handler": runpod_handler_sync,
                "return_aggregate_stream": True,  # Make results available via /run endpoint
            }
        )


# ASYNC MAIN BLOCK (commented out for now)
# if __name__ == "__main__":
#     import sys
#
#     if len(sys.argv) > 1 and sys.argv[1] == "--test":
#         # Run local test
#         test_handler_locally()
#     else:
#         # Start RunPod serverless
#         print("🚀 Starting RunPod Serverless Handler for Hunyuan 3D 2.1")
#         runpod.serverless.start(
#             {
#                 "handler": runpod_handler,
#                 "return_aggregate_stream": True,  # Make results available via /run endpoint
#             }
#         )
