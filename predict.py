from cog import BasePredictor, BaseModel, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
import shutil
from hy3dgen.shapegen import (
    FaceReducer, FloaterRemover, DegenerateFaceRemover,
    Hunyuan3DDiTFlowMatchingPipeline
)
from hy3dgen.texgen import Hunyuan3DPaintPipeline

MODEL_PATH = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0/model.tar"
# DELIGHT_URL = "https://weights.replicate.delivery/default/tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0/delight.tar"
# PAINT_URL = "https://weights.replicate.delivery/default/tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0/paint.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Output(BaseModel):
    mesh: Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        #make directory model_path
        os.makedirs(MODEL_PATH, exist_ok=True)

        # Download weights
        if not os.path.exists(MODEL_PATH + "/hunyuan3d-dit-v2-0"):
            download_weights(MODEL_URL, MODEL_PATH + "/hunyuan3d-dit-v2-0")
        # if not os.path.exists(MODEL_PATH + "/hunyuan3d-delight-v2-0"):
        #     download_weights(DELIGHT_URL, MODEL_PATH + "/hunyuan3d-delight-v2-0")
        # if not os.path.exists(MODEL_PATH + "/hunyuan3d-paint-v2-0"):
        #     download_weights(PAINT_URL, MODEL_PATH + "/hunyuan3d-paint-v2-0")

        # Load shape generation model
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_PATH+"/hunyuan3d-dit-v2-0")
        self.floater_remove_worker = FloaterRemover()
        self.degenerate_face_remove_worker = DegenerateFaceRemover()
        self.face_reduce_worker = FaceReducer()

        # Load background remover
        from hy3dgen.rembg import BackgroundRemover
        self.rmbg_worker = BackgroundRemover()

    def predict(
        self,
        image: Path = Input(
            description="Input image for generating 3D shape",
            default=None
        ),
        steps: int = Input(
            description="Number of inference steps",
            default=50,
            ge=20,
            le=50,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for generation",
            default=5.5,
            ge=1.0,
            le=20.0,
        ),
        seed: int = Input(
            description="Random seed for generation",
            default=1234
        ),
        octree_resolution: int = Input(
            description="Octree resolution for mesh generation",
            choices=[256, 384, 512],
            default=256
        ),
        remove_background: bool = Input(
            description="Whether to remove background from input image",
            default=True
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        # Clean up past runs, delete output directory
        if os.path.exists("output"):
            shutil.rmtree("output")
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Set random seed
        generator = torch.Generator()
        generator = generator.manual_seed(seed)

        # Process input image
        if image is not None:
            input_image = Image.open(str(image))
            if remove_background or input_image.mode == "RGB":
                input_image = self.rmbg_worker(input_image.convert('RGB'))
        else:
            raise ValueError("Image must be provided")

        # Save processed input image
        input_image.save("output/input.png")

        # Generate 3D mesh
        mesh = self.pipeline(
            image=input_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution
        )[0]

        # Post-process mesh
        mesh = self.floater_remove_worker(mesh)
        mesh = self.degenerate_face_remove_worker(mesh)
        mesh = self.face_reduce_worker(mesh)

        # Set mesh color to dark gray
        mesh.visual.face_colors = [64, 64, 64, 255]  # RGB + Alpha for dark gray

        # Export mesh
        output_path = Path("output/gray_mesh.glb")
        mesh.export(str(output_path), include_normals=False)

        # Verify file exists and is readable
        if not Path(output_path).exists():
            raise RuntimeError(f"Failed to generate mesh file at {output_path}")

        return Output(mesh=output_path)