from collections import defaultdict
from argparse import Namespace
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as T

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
from SoccerNet.Evaluation.utils_calibration import SoccerPitch
from random import choices
import string
import os
import shutil
import cv2
from tvcalib.cam_modules import SNProjectiveCamera


from tvcalib.module import TVCalibModule
from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
from sn_segmentation.src.custom_extremities import generate_class_synthesis, get_line_extremities
from tvcalib.sncalib_dataset import custom_list_collate
from tvcalib.utils.io import detach_dict, tensor2list
from tvcalib.utils.objects_3d import SoccerPitchLineCircleSegments, SoccerPitchSNCircleCentralSplit
from tvcalib.inference import InferenceDatasetCalibration, InferenceDatasetSegmentation, InferenceSegmentationModel
from tvcalib.inference import get_camera_from_per_sample_output
from tvcalib.utils import visualization_mpl_min as viz

def debug(df, args, object3dcpu):
    sample = df.iloc[0]

    image_id = Path(sample.image_id).stem
    print(f"{image_id=}")
    image = Image.open(args.images_path / sample.image_id).convert("RGB")
    image = T.functional.to_tensor(image)

    cam = get_camera_from_per_sample_output(sample, args.lens_dist)

    f"aov_deg={torch.rad2deg(cam.phi_dict['aov'])}, t={torch.stack([cam.phi_dict[k] for k in ['c_x', 'c_y', 'c_z']], dim=-1)}, pan_deg={torch.rad2deg(cam.phi_dict['pan'])} tilt_deg={torch.rad2deg(cam.phi_dict['tilt'])} roll_deg={torch.rad2deg(cam.phi_dict['roll'])}"

    print(cam, cam.str_lens_distortion_coeff(b=0) if args.lens_dist else "")


    points_line, points_circle = sample["points_line"], sample["points_circle"]

    if args.lens_dist:
        # we visualize annotated points and image after undistortion
        image = cam.undistort_images(image.unsqueeze(0).unsqueeze(0)).squeeze()
        # print(points_line.shape) # expected: (1, 1, 3, S, N)
        points_line = SNProjectiveCamera.static_undistort_points(points_line.unsqueeze(0).unsqueeze(0), cam).squeeze()
        points_circle = SNProjectiveCamera.static_undistort_points(points_circle.unsqueeze(0).unsqueeze(0), cam).squeeze()
    else:
        psi = None

    fig, ax = viz.init_figure(args.image_width, args.image_height)
    ax = viz.draw_image(ax, image)

    ax = viz.draw_reprojection(ax, object3dcpu, cam)
    ax = viz.draw_selected_points(
        ax,
        object3dcpu,
        points_line,
        points_circle,
        kwargs_outer={
            "zorder": 1000,
            "rasterized": False,
            "s": 500,
            "alpha": 0.3,
            "facecolor": "none",
            "linewidths": 3,
        },
        kwargs_inner={
            "zorder": 1000,
            "rasterized": False,
            "s": 50,
            "marker": ".",
            "color": "k",
            "linewidths": 4.0,
        },
    )
    dpi = 50

    ax.set_xlim(0, args.image_width)
    ax.set_ylim(args.image_height, 0)

    plt.savefig(args.output_dir / f"{image_id}.pdf", dpi=dpi)
    plt.savefig(args.output_dir / f"{image_id}.svg", dpi=dpi)
    plt.savefig(args.output_dir / f"{image_id}.png", dpi=dpi)

def run_inference_on_folder(folder_path, model_path = "data/segment_localization/train_59.pt", image_width = 1280, image_height = 720, debug_flag = False):
    print("image_width:", image_width)
    print("image_height:", image_height)
    args = Namespace(
        images_path=Path(folder_path),
        output_dir=Path("tmp"),
        checkpoint=model_path,
        gpu=True,
        nworkers=12,
        batch_size_seg=16,
        batch_size_calib=256,
        imaobject3dcpuge_height=image_width,
        optim_steps=2000,
        lens_dist=False,
        write_masks=False,
        image_height=image_height,
        image_width=image_width
    )
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    object3d = SoccerPitchLineCircleSegments(
        device=device, base_field=SoccerPitchSNCircleCentralSplit()
    )
    object3dcpu = SoccerPitchLineCircleSegments(
        device="cpu", base_field=SoccerPitchSNCircleCentralSplit()
    )

    lines_palette = [0, 0, 0]
    for line_class in SoccerPitch.lines_classes:
        lines_palette.extend(SoccerPitch.palette[line_class])

    fn_generate_class_synthesis = partial(generate_class_synthesis, radius=4)
    fn_get_line_extremities = partial(get_line_extremities, maxdist=30,
                                    width=455, height=256, num_points_lines=4, num_points_circles=8)

    dataset_seg = InferenceDatasetSegmentation(
        args.images_path, args.image_width, args.image_height
    )

    print("number of images:", len(dataset_seg))
    dataloader_seg = torch.utils.data.DataLoader(
        dataset_seg,
        batch_size=args.batch_size_seg,
        num_workers=12,
        shuffle=False,
        collate_fn=custom_list_collate,
    )

    model_seg = InferenceSegmentationModel(args.checkpoint, device)

    image_ids = []
    keypoints_raw = []
    (args.output_dir / "masks").mkdir(parents=True, exist_ok=True)
    for batch_dict in tqdm(dataloader_seg):
        # semantic segmentation
        # image_raw: [B, 3, image_height, image_width]
        # image: [B, 3, 256, 455]
        with torch.no_grad():
            sem_lines = model_seg.inference(batch_dict["image"].to(device))
        sem_lines = sem_lines.cpu().numpy().astype(np.uint8)  # [B, 256, 455]

        # point selection
        with Pool(args.nworkers) as p:
            skeletons_batch = p.map(fn_generate_class_synthesis, sem_lines)
            keypoints_raw_batch = p.map(fn_get_line_extremities, skeletons_batch)

        # write to file
        if args.write_masks:
            print("Write masks to file")
            for image_id, mask in zip(batch_dict["image_id"], sem_lines):
                mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
                mask.putpalette(lines_palette)
                mask.convert("RGB").save(args.output_dir / "masks" / image_id)

        image_ids.extend(batch_dict["image_id"])
        keypoints_raw.extend(keypoints_raw_batch)

    model_calib = TVCalibModule(
        object3d,
        get_cam_distr(1.96, args.batch_size_calib, 1),
        get_dist_distr(args.batch_size_calib, 1) if args.lens_dist else None,
        (args.image_height, args.image_width),
        args.optim_steps,
        device,
        log_per_step=False,
        tqdm_kwqargs=None,
    )

    dataset_calib = InferenceDatasetCalibration(
        keypoints_raw, args.image_width, args.image_height, object3d)
    dataloader_calib = torch.utils.data.DataLoader(
        dataset_calib, args.batch_size_calib, collate_fn=custom_list_collate)

    per_sample_output = defaultdict(list)
    per_sample_output["image_id"] = [[x] for x in image_ids]
    for x_dict in dataloader_calib:
        _batch_size = x_dict["lines__ndc_projected_selection_shuffled"].shape[0]

        points_line = x_dict["lines__px_projected_selection_shuffled"]
        points_circle = x_dict["circles__px_projected_selection_shuffled"]
        print(f"{points_line.shape=}, {points_circle.shape=}")

        per_sample_loss, cam, _ = model_calib.self_optim_batch(x_dict)
        output_dict = tensor2list(detach_dict(
            {**cam.get_parameters(_batch_size), **per_sample_loss}))

        output_dict["points_line"] = points_line
        output_dict["points_circle"] = points_circle
        for k in output_dict.keys():
            per_sample_output[k].extend(output_dict[k])

    df = pd.DataFrame.from_dict(per_sample_output)

    df = df.explode(
        column=[k for k, v in per_sample_output.items() if isinstance(v, list)])
    df.set_index("image_id", inplace=True, drop=False)

    sample = df.iloc[0]
    cam = get_camera_from_per_sample_output(sample, args.lens_dist)

    if debug_flag:
        debug(df, args, object3dcpu)

    return cam, output_dict

def run_inference_on_image(image, model_path = "data/segment_localization/train_59.pt", debug = False):
    folder_name = 'temp_'+''.join(choices(
        string.ascii_uppercase + string.digits, k=5))
    
    shutil.rmtree(folder_name, ignore_errors=True)
    
    os.mkdir(folder_name)
    image_path = os.path.join(folder_name, 'image.jpg')
    cv2.imwrite(image_path, image)
    
    res = run_inference_on_folder(folder_name, model_path, image.shape[1], image.shape[0], debug_flag = debug)

    os.remove(image_path)
    os.rmdir(folder_name)
    return res
