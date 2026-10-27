"""Run Tencent's actual Hunyuan3D-Paint 2.1 PBR pipeline on an existing GLB.

Upstream mesh rendering writes Wavefront OBJ and optionally converts that OBJ
to a GLB using bpy; passing a .glb as its output_mesh_path instead writes
OBJ *text* with a .glb extension and may attempt to import it as OBJ.  The
adapter always gives upstream an .obj staging path and validates the produced
binary GLB before atomic publication of the final output.

Run with the separate Paint Python and vendor/Hunyuan3D-2.1 as cwd.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from i3d.artifacts import obj_intermediate_for_glb, validate_glb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mesh", "image", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--resolution", type=int, choices=(512, 768), default=512)
    parser.add_argument("--views", type=int, choices=(6, 7, 8, 9), default=6)
    args = parser.parse_args()

    source = validate_glb(args.mesh.resolve())
    image = args.image.resolve()
    if not image.is_file() or image.stat().st_size == 0:
        raise FileNotFoundError(f"Missing reference image: {image}")
    destination = args.output.resolve()
    temp_obj = obj_intermediate_for_glb(
        destination.with_name(destination.stem + ".i3d-working.glb")
    )
    temp_glb = temp_obj.with_suffix(".glb")
    # The output must be created by THIS run, not left by a failed prior run.
    temp_glb.unlink(missing_ok=True)

    paint_root = ROOT / "vendor" / "Hunyuan3D-2.1"
    if not (paint_root / "hy3dpaint" / "textureGenPipeline.py").is_file():
        raise FileNotFoundError(f"Missing original Tencent Paint source: {paint_root}")
    os.chdir(paint_root)
    sys.path.insert(0, str(paint_root / "hy3dpaint"))
    sys.path.insert(0, str(paint_root))
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    config = Hunyuan3DPaintConfig(args.views, args.resolution)
    config.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    config.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    config.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    if not Path(config.realesrgan_ckpt_path).is_file():
        raise FileNotFoundError("Missing Hunyuan Paint RealESRGAN checkpoint")
    if not Path(config.multiview_cfg_path).is_file():
        raise FileNotFoundError("Missing original Hunyuan Paint model configuration")

    print("HUNYUAN_PAINT_INPUT", source, flush=True)
    print("HUNYUAN_PAINT_STAGING_OBJ", temp_obj, flush=True)
    pipeline = Hunyuan3DPaintPipeline(config)
    result_obj = Path(pipeline(
        mesh_path=str(source),
        image_path=str(image),
        output_mesh_path=str(temp_obj),
        save_glb=True,
    )).resolve()
    if result_obj != temp_obj:
        raise RuntimeError(f"Unexpected upstream Paint OBJ output: {result_obj}")
    if not result_obj.is_file():
        raise FileNotFoundError(f"Upstream Paint did not produce its OBJ: {result_obj}")
    validate_glb(temp_glb)  # Detect silent upstream convert_obj_to_glb(False).
    os.replace(temp_glb, destination)
    validate_glb(destination)
    print("HUNYUAN_PBR_GLB_READY", destination, destination.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
