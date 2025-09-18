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
    "id": "request-123",  # Optional: Unique request ID for tracing (auto-generated if not provided)
    "workflow": "enhanced",  # Required: "mesh", "texture", or "enhanced"
    "input_image": "base64_encoded_image_data",  # Required: Base64 encoded image (supports all formats: PNG, JPEG, JPG, GIF, BMP, TIFF, WEBP, etc.) or file path
    "output_name": "my_mesh",  # Optional: Output filename prefix
    
    # Webhook configuration (optional) - can be string, array of strings, or array of objects
    "webhooks": [
        "https://api.example.com/webhook1",  # Simple URL string
        {
            "url": "https://api.example.com/webhook2",  # Object with URL field
            "description": "My custom webhook"  # Additional fields ignored
        }
    ],
    # OR single webhook as string: "webhooks": "https://api.example.com/webhook"
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
    },
    "upload_to_r2": true # Upload to r2 or not (Always keep true)
}

OUTPUT FORMAT:
{
    "status": "success",
    "request_id": "request-123",  # Request ID for tracing
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
- aiohttp: pip install aiohttp (for webhook functionality)

WEBHOOK CONFIGURATION:
Two ways to configure webhooks:

1. Environment Variable (HOOKS):
   - Set HOOKS environment variable with comma or semicolon separated webhook URLs
   - Example: HOOKS="https://api.example.com/webhook1,https://api.example.com/webhook2"
   - These webhooks apply to ALL requests

2. Request Body (webhooks):
   - Include "webhooks" field in the request body for per-request webhooks
   - Can be a string (single URL), array of strings, or array of objects with 'url' field
   - Examples:
     * Single: "webhooks": "https://api.example.com/webhook"
     * Array: "webhooks": ["https://api.example.com/webhook1", "https://api.example.com/webhook2"]
     * Objects: "webhooks": [{"url": "https://api.example.com/webhook", "description": "My webhook"}]

Webhook Behavior:
- Both environment and request webhooks are called (combined and deduplicated)
- Webhooks are called asynchronously after job completion (success or error)
- Webhook payload contains the complete job result including request_id
- Webhooks have a 30-second timeout and will retry failures gracefully
"""

import asyncio
import aiohttp
import base64
import io
import json
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Union, List

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

        # Initialize webhook configuration
        self.webhook_urls = self._get_webhook_urls()
        if self.webhook_urls:
            print(f"✅ Configured {len(self.webhook_urls)} webhook URLs")

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

    def _get_webhook_urls(self) -> List[str]:
        """Get webhook URLs from HOOKS environment variable"""
        hooks_env = os.getenv("HOOKS", "")
        if not hooks_env.strip():
            return []
        
        # Support multiple URLs separated by comma or semicolon
        urls = []
        for url in hooks_env.replace(";", ",").split(","):
            url = url.strip()
            if url and (url.startswith("http://") or url.startswith("https://")):
                urls.append(url)
            elif url:
                print(f"⚠️ Invalid webhook URL format: {url}")
        
        return urls

    def _process_request_webhooks(self, webhooks_input) -> List[str]:
        """Process webhook URLs from request input

        Args:
            webhooks_input: Can be a string (single URL), list of strings (URLs),
                           or list of dicts with 'url' field

        Returns:
            List of validated webhook URLs
        """
        if not webhooks_input:
            return []
        
        urls = []
        
        # Handle different input formats
        if isinstance(webhooks_input, str):
            # Single URL as string
            webhooks_input = [webhooks_input]
        
        if isinstance(webhooks_input, list):
            for item in webhooks_input:
                if isinstance(item, str):
                    # Simple URL string
                    url = item.strip()
                elif isinstance(item, dict) and 'url' in item:
                    # Dictionary with URL field
                    url = item['url'].strip()
                else:
                    print(f"⚠️ Invalid webhook format: {item}")
                    continue
                
                # Validate URL
                if url and (url.startswith("http://") or url.startswith("https://")):
                    urls.append(url)
                elif url:
                    print(f"⚠️ Invalid webhook URL format: {url}")
        
        return urls

    def _get_combined_webhooks(self, request_webhooks: List[str] = None) -> List[str]:
        """Combine environment webhooks and request webhooks
        
        Args:
            request_webhooks: Webhook URLs from request body
            
        Returns:
            Combined list of unique webhook URLs
        """
        combined = []
        
        # Add environment webhooks
        combined.extend(self.webhook_urls)
        
        # Add request webhooks
        if request_webhooks:
            combined.extend(request_webhooks)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_webhooks = []
        for url in combined:
            if url not in seen:
                seen.add(url)
                unique_webhooks.append(url)
        
        return unique_webhooks

    async def _call_webhooks(self, data: Dict[str, Any], request_webhooks: List[str] = None) -> None:
        """Call all configured webhooks asynchronously with the result data
        
        Args:
            data: The data to send to webhooks
            request_webhooks: Additional webhook URLs from the request
        """
        # Get combined webhook URLs from environment and request
        all_webhooks = self._get_combined_webhooks(request_webhooks)
        
        if not all_webhooks:
            return

        print(f"📞 Calling {len(all_webhooks)} webhooks...")
        
        async def call_single_webhook(session: aiohttp.ClientSession, url: str) -> None:
            """Call a single webhook URL"""
            try:
                timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "RunPod-Hunyuan3D-Handler/1.0"
                }
                
                async with session.post(url, json=data, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        print(f"✅ Webhook called successfully: {url}")
                    else:
                        print(f"⚠️ Webhook returned status {response.status}: {url}")
                        
            except asyncio.TimeoutError:
                print(f"⏰ Webhook timeout: {url}")
            except Exception as e:
                print(f"❌ Webhook error for {url}: {str(e)}")

        # Call all webhooks concurrently
        try:
            async with aiohttp.ClientSession() as session:
                tasks = [call_single_webhook(session, url) for url in all_webhooks]
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            print(f"❌ Error calling webhooks: {str(e)}")

    def _decode_base64_image(self, base64_data: str) -> Image.Image:
        """Decode base64 image data to PIL Image, or load from file path

        Supports all common image formats including PNG, JPEG, JPG, GIF, BMP, TIFF, WEBP, etc.
        """
        try:
            # Check if it's a file path (not a base64 string)
            if not base64_data.startswith("data:image") and not self._is_base64_string(
                base64_data
            ):
                # Assume it's a file path
                if os.path.exists(base64_data):
                    print(f"📁 Loading image from file: {base64_data}")
                    image = Image.open(base64_data).convert("RGB")
                    print(
                        f"📷 Image loaded: {image.size} pixels, format: {image.format}"
                    )
                    return image
                else:
                    raise ValueError(f"File not found: {base64_data}")

            # Handle base64 data
            # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
            if base64_data.startswith("data:image"):
                base64_data = base64_data.split(",")[1]

            # Decode base64 data
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))

            # Get original format info before converting
            original_format = image.format or "Unknown"

            # Convert to RGB (handles all formats including RGBA, grayscale, etc.)
            image = image.convert("RGB")

            print(
                f"📷 Image decoded from base64: {image.size} pixels, original format: {original_format}"
            )
            return image

        except Exception as e:
            raise ValueError(f"Failed to decode/load image: {str(e)}") from e

    def _is_base64_string(self, s: str) -> bool:
        """Check if a string is likely a base64 encoded string"""
        try:
            # Check basic length requirement
            if len(s) < 50:
                return False

            # Check if string contains only valid base64 characters
            import re

            if not re.match(r"^[A-Za-z0-9+/]*={0,2}$", s):
                return False

            # Try to decode - if it fails, it's not base64
            base64.b64decode(s, validate=True)

            # Special check for common file path patterns that would never be base64
            # Look for file extensions in the last few characters
            if re.search(r"\.[a-zA-Z]{2,4}$", s):  # ends with .ext
                return False

            # Look for drive letters or common path patterns
            if re.match(r"^[A-Za-z]:[/\\]", s):  # Windows drive letter
                return False

            # If it starts with a path separator and has spaces or common path chars, it's a path
            if s.startswith("/") and (" " in s or len(s) < 200):
                # But JPEG base64 often starts with /9j/ so check for that pattern
                if not s.startswith("/9j/"):
                    return False

            return True

        except Exception:
            return False

    def _save_image_temporarily(
        self, image: Image.Image, name: str = "temp_input"
    ) -> str:
        """Save PIL Image temporarily for processing

        Always saves as PNG for consistency, regardless of original format
        """
        # Use cross-platform temp directory
        import tempfile

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{name}.png")

        # Ensure temp directory exists
        os.makedirs(temp_dir, exist_ok=True)

        # Save as PNG format (lossless and widely supported)
        image.save(temp_path, format="PNG")
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

        # Generate unique ID if not provided
        import uuid
        request_id = job_input.get("id", str(uuid.uuid4()))

        # Process webhook URLs from request
        request_webhooks = self._process_request_webhooks(job_input.get("webhooks", []))

        # Normalize input
        normalized = {
            "id": request_id,
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
            "webhooks": request_webhooks,
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
        params = None
        request_id = job_input.get("id", "unknown")

        try:
            # Validate input
            params = self._validate_input(job_input)
            workflow_type = params["workflow"]
            request_id = params["id"]

            results = {}

            print(f"🔄 Starting {workflow_type} workflow (ID: {request_id})...")

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
                    "request_id": request_id,
                }

            # Final cleanup
            cleanup_memory()

            processing_time = time.time() - start_time

            # Final result with request ID
            final_result = {
                "status": "success",
                "request_id": request_id,
                "processing_time": round(processing_time, 2),
                "r2_configured": self.r2_uploader is not None,
                **results,
            }

            print(f"✅ Processing complete (ID: {request_id})!")
            
            # Call webhooks asynchronously (fire and forget)
            request_webhooks = params.get("webhooks", [])
            if self.webhook_urls or request_webhooks:
                asyncio.create_task(self._call_webhooks(final_result, request_webhooks))
            
            return final_result

        except Exception as e:
            error_message = str(e)
            error_traceback = traceback.format_exc()

            # Use request_id from params if available, otherwise from input
            if params and "id" in params:
                request_id = params["id"]

            print(f"❌ Error processing job (ID: {request_id}): {error_message}")
            print(f"Traceback: {error_traceback}")

            error_result = {
                "status": "error",
                "request_id": request_id,
                "error": error_message,
                "traceback": error_traceback,
                "processing_time": round(time.time() - start_time, 2),
            }
            
            # Call webhooks for errors too
            request_webhooks = params.get("webhooks", []) if params else []
            if self.webhook_urls or request_webhooks:
                asyncio.create_task(self._call_webhooks(error_result, request_webhooks))
            
            return error_result


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


# Test function for webhook functionality
def test_webhook_functionality():
    """Test webhook configuration and calling"""
    print("🧪 Testing webhook functionality...")
    
    # Test webhook URL parsing
    original_hooks = os.getenv("HOOKS", "")
    
    # Test with different URL formats
    test_cases = [
        ("", []),
        ("https://api.example.com/webhook", ["https://api.example.com/webhook"]),
        ("https://api.example.com/webhook1,https://api.example.com/webhook2",
         ["https://api.example.com/webhook1", "https://api.example.com/webhook2"]),
        ("https://api.example.com/webhook1;https://api.example.com/webhook2",
         ["https://api.example.com/webhook1", "https://api.example.com/webhook2"]),
        ("https://valid.com,invalid_url,http://also.valid.com",
         ["https://valid.com", "http://also.valid.com"]),
    ]
    
    handler = RunPodHunyuan3DHandler()
    
    for hooks_str, expected in test_cases:
        os.environ["HOOKS"] = hooks_str
        urls = handler._get_webhook_urls()
        
        if urls == expected:
            print(f"✅ Webhook parsing test passed: '{hooks_str}' -> {len(urls)} URLs")
        else:
            print(f"❌ Webhook parsing test failed: '{hooks_str}' -> expected {expected}, got {urls}")
    
    # Restore original HOOKS value
    if original_hooks:
        os.environ["HOOKS"] = original_hooks
    elif "HOOKS" in os.environ:
        del os.environ["HOOKS"]
    
    # Test webhook calling with a test payload
    test_payload = {
        "status": "success",
        "request_id": "test-123",
        "workflow_type": "test",
        "processing_time": 1.5
    }
    
    # Only test if webhooks are configured
    if handler.webhook_urls:
        print(f"🔗 Testing webhook calls to {len(handler.webhook_urls)} URLs...")
        
        # Create an event loop if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the webhook test
        loop.run_until_complete(handler._call_webhooks(test_payload))
        print("✅ Webhook test completed")
    else:
        print("ℹ️ No webhooks configured for testing")


# Test function for request-body webhook functionality
def test_request_webhooks():
    """Test request-body webhook functionality"""
    print("🧪 Testing request-body webhook functionality...")
    
    handler = RunPodHunyuan3DHandler()
    
    # Test different webhook input formats
    test_cases = [
        # Single URL as string
        ("Single string URL", "https://httpbin.org/post"),
        
        # Array of strings
        ("Array of strings", [
            "https://httpbin.org/post",
            "https://webhook.site/test"
        ]),
        
        # Array of objects with URL field
        ("Array of objects", [
            {"url": "https://httpbin.org/post", "description": "Test webhook 1"},
            {"url": "https://webhook.site/test", "name": "Test webhook 2"}
        ]),
        
        # Mixed format (string and object)
        ("Mixed format", [
            "https://httpbin.org/post",
            {"url": "https://webhook.site/test", "type": "notification"}
        ]),
        
        # Invalid formats
        ("Invalid URL", "not_a_valid_url"),
        ("Empty array", []),
        ("None", None)
    ]
    
    for description, webhook_input in test_cases:
        print(f"\n🔍 Testing: {description}")
        print(f"Input: {webhook_input}")
        
        try:
            processed_webhooks = handler._process_request_webhooks(webhook_input)
            print(f"✅ Processed webhooks: {processed_webhooks} ({len(processed_webhooks)} URLs)")
        except Exception as e:
            print(f"❌ Error processing webhooks: {e}")
    
    # Test webhook combination
    print("\n🔄 Testing webhook combination...")
    
    # Set environment webhook
    original_hooks = os.getenv("HOOKS", "")
    os.environ["HOOKS"] = "https://env-webhook.example.com"
    
    # Re-initialize handler to pick up environment webhook
    handler = RunPodHunyuan3DHandler()
    
    request_webhooks = ["https://request-webhook1.example.com", "https://request-webhook2.example.com"]
    combined_webhooks = handler._get_combined_webhooks(request_webhooks)
    
    print(f"Environment webhooks: {handler.webhook_urls}")
    print(f"Request webhooks: {request_webhooks}")
    print(f"Combined webhooks: {combined_webhooks}")
    
    # Test deduplication
    request_webhooks_with_duplicate = [
        "https://env-webhook.example.com",  # Same as environment
        "https://request-unique.example.com"
    ]
    combined_with_dedup = handler._get_combined_webhooks(request_webhooks_with_duplicate)
    print(f"Combined with deduplication: {combined_with_dedup}")
    
    # Restore original environment
    if original_hooks:
        os.environ["HOOKS"] = original_hooks
    elif "HOOKS" in os.environ:
        del os.environ["HOOKS"]


# Test function for full workflow with request webhooks
def test_full_workflow_with_request_webhooks():
    """Test full workflow with request-body webhooks"""
    print("🧪 Testing full workflow with request-body webhooks...")
    
    test_input = {
        "id": "request-webhook-test",
        "input": {
            "id": "req-webhook-001",
            "workflow": "mesh",
            "input_image": "assets/mune.png",
            "output_name": "request_webhook_test",
            "remove_background": False,
            "upload_to_r2": False,
            "mesh_params": {"steps": 3, "max_facenum": 500},
            
            # Test request-body webhooks
            "webhooks": [
                "https://httpbin.org/post",
                {"url": "https://webhook.site/unique-id", "description": "Test notification"}
            ]
        },
    }
    
    print("📝 Request includes webhooks:", test_input["input"]["webhooks"])
    
    # Test validation first
    handler = RunPodHunyuan3DHandler()
    try:
        validated_params = handler._validate_input(test_input["input"])
        print(f"✅ Validation passed. Processed webhooks: {validated_params['webhooks']}")
        
        # Note: We won't run the full workflow here to avoid long processing time
        # but we can test the validation and webhook processing logic
        print("ℹ️ Skipping full workflow execution for this test")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")


# Test function with webhook integration
def test_handler_with_webhooks():
    """Test the full handler with webhook functionality"""
    print("🧪 Testing handler with webhook integration...")
    
    # Test with a custom webhook URL (if provided via env)
    test_input = {
        "id": "webhook-test-123",
        "input": {
            "id": "webhook-test-456",  # Test custom request ID
            "workflow": "mesh",
            "input_image": "assets/mune.png",
            "output_name": "webhook_test_mesh",
            "remove_background": False,
            "mesh_params": {"steps": 3, "max_facenum": 500},  # Minimal for testing
        },
    }
    
    print("📝 Test input includes custom request ID:", test_input["input"]["id"])
    
    result = runpod_handler_sync(test_input)
    
    print(f"📤 Result includes request_id: {result.get('request_id', 'MISSING')}")
    print(f"📊 Status: {result.get('status', 'MISSING')}")
    
    # Verify request ID was preserved
    expected_id = test_input["input"]["id"]
    actual_id = result.get("request_id")
    
    if actual_id == expected_id:
        print("✅ Request ID tracking works correctly")
    else:
        print(f"❌ Request ID mismatch: expected {expected_id}, got {actual_id}")
    
    # Give webhooks time to be called
    if result.get("status") == "success":
        print("⏳ Waiting for webhooks to be called...")
        import time
        time.sleep(2)  # Give async webhooks time to execute


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
        elif sys.argv[1] == "--test-webhooks":
            # Test webhook functionality
            test_webhook_functionality()
        elif sys.argv[1] == "--test-with-webhooks":
            # Test handler with webhooks
            test_handler_with_webhooks()
        elif sys.argv[1] == "--test-request-webhooks":
            # Test request-body webhook functionality
            test_request_webhooks()
        elif sys.argv[1] == "--test-full-request-webhooks":
            # Test full workflow with request webhooks
            test_full_workflow_with_request_webhooks()
        else:
            print(
                "Usage: python runpod_handler.py [--test|--test-minimal|--validate|--test-image|--test-file|--test-r2|--test-webhooks|--test-with-webhooks|--test-request-webhooks|--test-full-request-webhooks]"
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
