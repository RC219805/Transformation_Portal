#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch panoptic segmentation mask generator for depth_tools.py integration.

Generates *_mask_sky.png and *_mask_building.png from Detectron2 panoptic outputs.
Usage:
  python run_detectron2_panoptic_batch.py \
      --images-root "/Users/rc/Desktop/my_project/outputs" \
      --depths-root "/Users/rc/Desktop/my_project/outputs/depth/750_Picacho" \
      --mask-root   "/Users/rc/Desktop/my_project/outputs/seg/750_Picacho" \
      --device cpu --save-panoptic
"""
import os
import glob
import numpy as np
from PIL import Image
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

def build_predictor(device="cpu"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    )
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return DefaultPredictor(cfg), MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

def extract_masks(panoptic_seg, segments_info, meta):
    """
    Create binary masks for sky and building categories.
    """
    id_to_name = {s["id"]: meta.stuff_classes[s["category_id"]] for s in segments_info}
    sky_ids = [sid for sid, name in id_to_name.items() if "sky" in name.lower()]
    bld_ids = [sid for sid, name in id_to_name.items()
               if any(x in name.lower() for x in ["building", "wall", "house", "structure"])]

    sky_mask = np.isin(panoptic_seg.cpu().numpy(), sky_ids).astype(np.uint8) * 255
    bld_mask = np.isin(panoptic_seg.cpu().numpy(), bld_ids).astype(np.uint8) * 255
    return sky_mask, bld_mask

def main(images_root, depths_root, mask_root, device="cpu", save_panoptic=False):
    os.makedirs(mask_root, exist_ok=True)
    predictor, meta = build_predictor(device=device)

    depth_maps = sorted(glob.glob(os.path.join(depths_root, "*_depth16.png")))
    print(f"Found {len(depth_maps)} depth maps in {depths_root}")
    for i, dp in enumerate(depth_maps, 1):
        base = os.path.basename(dp).replace("_depth16.png", "")
        # Find matching enhanced image
        pats = [f"{base}_ENH*", f"{base}_PUNCHY*", f"{base}_AGX*", f"{base}_GOLDEN*"]
        imgs = []
        for pat in pats:
            imgs.extend(glob.glob(os.path.join(images_root, "**", pat), recursive=True))
        if not imgs:
            print(f"[{i}] skip (no image): {base}")
            continue
        src = sorted(imgs)[0]

        im = np.asarray(Image.open(src).convert("RGB"))
        outputs = predictor(im)
        panoptic_seg, segments_info = outputs["panoptic_seg"]
        sky_mask, bld_mask = extract_masks(panoptic_seg, segments_info, meta)

        Image.fromarray(sky_mask).save(os.path.join(mask_root, f"{base}_mask_sky.png"))
        Image.fromarray(bld_mask).save(os.path.join(mask_root, f"{base}_mask_building.png"))

        if save_panoptic:
            vis = Visualizer(im[:, :, ::-1], meta, instance_mode=ColorMode.IMAGE_BW)
            vis_out = vis.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
            Image.fromarray(vis_out.get_image()[:, :, ::-1]).save(
                os.path.join(mask_root, f"{base}_panoptic_vis.jpg"))
        print(f"[{i}] ✓ {base}")

    print(f"\nDone → {mask_root}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--depths-root", required=True)
    ap.add_argument("--mask-root", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--save-panoptic", action="store_true")
    args = ap.parse_args()
    main(args.images_root, args.depths_root, args.mask_root,
         device=args.device, save_panoptic=args.save_panoptic)
