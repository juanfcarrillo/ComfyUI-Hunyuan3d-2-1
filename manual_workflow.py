from PIL import Image


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
    pil2tensor,
)


class ManualHunyuan3DWorkflow:
    def __init__(self):
        # Initialize the node instances
        self.vae_loader = Hy3D21VAELoader()
        self.mesh_generator = Hy3DMeshGenerator()
        self.vae_decoder = Hy3D21VAEDecode()
        self.postprocess_mesh = Hy3D21PostprocessMesh()
        self.export_mesh = Hy3D21ExportMesh()

    def load_image(self, image_path):
        """Load image and convert to tensor format expected by the nodes"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path

        # Convert PIL to tensor format (add batch dimension)
        image_tensor = pil2tensor(image)
        return image_tensor

    def get_number_of_faces(self, trimesh):
        """Get the number of faces from a trimesh object"""
        return trimesh.faces.shape[0]

    def run_workflow(
        self,
        vae_model_name,
        diffusion_model_name,
        input_image_path,
        output_mesh_name="generated_mesh",
        # VAE Loader parameters
        vae_config=None,
        # Mesh Generator parameters
        steps=50,
        guidance_scale=5.0,
        seed=0,
        attention_mode="sdpa",
        # VAE Decoder parameters
        box_v=1.01,
        octree_resolution=384,
        num_chunks=8000,
        mc_level=0,
        mc_algo="mc",
        enable_flash_vdm=True,
        force_offload=False,
        # Post Process parameters
        remove_floaters=True,
        remove_degenerate_faces=True,
        reduce_faces=True,
        max_facenum=40000,
        smooth_normals=False,
        # Export parameters
        file_format="glb",
        save_file=True,
    ):

        print("Starting Hunyuan 3D 2.1 Manual Workflow...")

        # Step 1: Load VAE model
        print("1. Loading VAE model...")
        vae = self.vae_loader.loadmodel(vae_model_name, vae_config)[0]
        print(f"   VAE model loaded: {vae_model_name}")

        # Step 2: Load and prepare input image
        print("2. Loading input image...")
        image_tensor = self.load_image(input_image_path)
        print(f"   Image loaded from: {input_image_path}")
        print(f"   Image shape: {image_tensor.shape}")

        # Step 3: Generate mesh latents
        print("3. Generating mesh latents...")
        latents = self.mesh_generator.loadmodel(
            model=diffusion_model_name,
            image=image_tensor,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            attention_mode=attention_mode,
        )[0]
        print(f"   Latents generated with {steps} steps")

        # Step 4: Decode latents to mesh
        print("4. Decoding latents to mesh...")
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

        # Step 7: Export mesh
        print("7. Exporting mesh...")
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
        }

        print("Workflow completed successfully!")
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

    def load_image(self, image_path):
        """Load image and convert to tensor format expected by the nodes"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path

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
        # MultiViews Generator parameters
        view_size=1024,
        steps=35,
        guidance_scale=5.6,
        texture_size=4096,
        normal_texture=True,
        unwrap_mesh=True,
        save_after_generate=False,
        correct_after_generate="randomize",
        # Bake MultiViews parameters
        albedo_texture=True,
        mr_texture=True,
        # InPaint parameters
        vertex_inpaint=True,
        method="NS",
        # Upscale parameters
        upscale_model_name="RealESRGAN_x4plus.pth",
        upscale_albedo=True,
        upscale_mr=True,
        # Export parameters
        file_format="glb",
        save_file=True,
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
        image_tensor = self.load_image(input_image_path)
        print(f"   Image loaded from: {input_image_path}")
        print(f"   Image shape: {image_tensor.shape}")

        # Step 4: Generate multi-views
        print("4. Generating multi-view textures...")
        multiview_results = self.multiviews_generator.genmultiviews(
            trimesh=uv_wrapped_mesh,
            camera_config=camera_config,
            view_size=view_size,
            image=image_tensor,
            steps=steps,
            guidance_scale=guidance_scale,
            texture_size=texture_size,
            unwrap_mesh=unwrap_mesh,
            seed=200434251488993,
        )

        pipeline = multiview_results[0]
        albedo = multiview_results[1]
        normals = multiview_results[2]
        images = multiview_results[4]

        print("   Multi-view generation completed")
        print(f"   Generated {len(images)} view images")

        # Step 5: Bake multi-views into textures
        print("5. Baking multi-views into textures...")
        bake_results = self.bake_multiviews.process(
            pipeline=pipeline,
            camera_config=camera_config,
            albedo=albedo,
            mr=normals,  # Using normals as metallic-roughness input
        )

        baked_pipeline = bake_results[0]
        baked_albedo = bake_results[1]
        albedo_mask = bake_results[2]
        baked_mr = bake_results[3] if len(bake_results) > 3 else None
        mr_mask = bake_results[4] if len(bake_results) > 4 else None

        print("   Multi-view baking completed")

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
        inpaint_results = self.inpaint.process(
            pipeline=baked_pipeline,
            albedo=final_albedo,
            albedo_mask=albedo_mask,
            mr=final_mr if final_mr is not None else final_albedo,
            mr_mask=mr_mask if mr_mask is not None else albedo_mask,
            output_mesh_name=output_mesh_name,
        )

        inpainted_albedo = inpaint_results[0]
        output_glb_path = inpaint_results[1]

        print("   Texture inpainting completed")

        # Step 10: Export final textured mesh
        print("10. Exporting final textured mesh...")
        final_output_path = self.export_mesh.process(
            trimesh=uv_wrapped_mesh,  # Use the UV-wrapped mesh
            filename_prefix=f"3D/{output_mesh_name}_final",
            file_format=file_format,
            save_file=save_file,
        )[0]
        print(f"    Final textured mesh exported to: {final_output_path}")

        # Return results
        results = {
            "uv_wrapped_mesh": uv_wrapped_mesh,
            "camera_config": camera_config,
            "multiview_images": images,
            "baked_albedo": baked_albedo,
            "baked_mr": baked_mr,
            "albedo_mask": albedo_mask,
            "mr_mask": mr_mask,
            "final_albedo": final_albedo,
            "final_mr": final_mr,
            "inpainted_albedo": inpainted_albedo,
            "output_glb_path": output_glb_path,
            "final_output_path": final_output_path,
            "output_mesh_name": output_mesh_name,
        }

        print("Texture generation workflow completed successfully!")
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
    ):

        print("Starting Complete Hunyuan 3D 2.1 Workflow (Mesh + Texture)...")

        # Default parameters
        if mesh_params is None:
            mesh_params = {
                "steps": 50,
                "guidance_scale": 5.0,
                "seed": 42,
                "max_facenum": 40000,
            }

        if texture_params is None:
            texture_params = {
                "view_size": 1024,
                "steps": 35,
                "guidance_scale": 5.6,
                "upscale_albedo": True,
                "upscale_mr": True,
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


def main():
    """Example usage of the complete workflow"""
    import argparse

    parser = argparse.ArgumentParser(description="Hunyuan 3D 2.1 Manual Workflow")
    parser.add_argument(
        "--workflow",
        choices=["mesh", "texture", "complete"],
        default="mesh",
        help="Which workflow to run",
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
                "steps": 50,
                "guidance_scale": 5.0,
                "seed": 42,
                "max_facenum": 40000,
                "file_format": "glb",
                "save_file": True,
            }
            results = workflow.run_workflow(**config)

            # Print summary
            print("\n" + "=" * 50)
            print("WORKFLOW SUMMARY")
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
            print("Texture-only workflow requires an existing mesh.")
            print("Please provide a trimesh object or run the complete workflow.")

        elif args.workflow == "complete":
            # Run complete workflow
            workflow = CompleteHunyuan3DWorkflow()
            results = workflow.run_complete_workflow(
                vae_model_name=args.vae_model,
                diffusion_model_name=args.diffusion_model,
                input_image_path=args.input_image,
                output_mesh_name=args.output_name,
            )
            print("\nComplete workflow finished!")
            print(f"Final textured mesh: {results['final_mesh_path']}")

    except Exception as e:
        print(f"Error running workflow: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
