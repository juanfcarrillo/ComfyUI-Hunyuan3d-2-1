from PIL import Image


# Import the node classes from the nodes.py file
from nodes import (
    Hy3D21VAELoader,
    Hy3DMeshGenerator,
    Hy3D21VAEDecode,
    Hy3D21PostprocessMesh,
    Hy3D21ExportMesh,
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


def main():
    """Example usage of the manual workflow"""

    # Initialize workflow
    workflow = ManualHunyuan3DWorkflow()

    # Configure parameters
    config = {
        "vae_model_name": "model.fp16.ckpt",  # Update with actual VAE model name
        "diffusion_model_name": "model.fp16.ckpt",  # Update with actual diffusion model name
        "input_image_path": "assets/mune.png",  # Update with actual image path
        "output_mesh_name": "my_generated_mesh",
        # Generation parameters
        "steps": 50,
        "guidance_scale": 5.0,
        "seed": 42,
        "attention_mode": "sdpa",
        # Decoder parameters
        "box_v": 1.01,
        "octree_resolution": 384,
        "num_chunks": 8000,
        "mc_level": 0,
        "mc_algo": "mc",
        "enable_flash_vdm": True,
        # Post-processing parameters
        "remove_floaters": True,
        "remove_degenerate_faces": True,
        "reduce_faces": True,
        "max_facenum": 40000,
        "smooth_normals": False,
        # Export parameters
        "file_format": "glb",
        "save_file": True,
    }

    try:
        # Run the workflow
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

    except Exception as e:
        print(f"Error running workflow: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
