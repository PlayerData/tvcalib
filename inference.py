from collections import defaultdict
from argparse import Namespace
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
from SoccerNet.Evaluation.utils_calibration import SoccerPitch


from tvcalib.module import TVCalibModule
from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
from sn_segmentation.src.custom_extremities import generate_class_synthesis, get_line_extremities
from tvcalib.sncalib_dataset import custom_list_collate
from tvcalib.utils.io import detach_dict, tensor2list
from tvcalib.utils.objects_3d import SoccerPitchLineCircleSegments, SoccerPitchSNCircleCentralSplit
from tvcalib.inference import InferenceDatasetCalibration, InferenceDatasetSegmentation, InferenceSegmentationModel
from tvcalib.inference import get_camera_from_per_sample_output

def run_inference(image_path):
    args = Namespace(
        images_path=Path(image_path),
        output_dir=Path("tmp"),
        checkpoint="data/segment_localization/train_59.pt",
        gpu=True,
        nworkers=12,
        batch_size_seg=16,
        batch_size_calib=256,
        image_width=1280,
        image_height=720,
        optim_steps=2000,
        lens_dist=False,
        write_masks=False
    )
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    object3d = SoccerPitchLineCircleSegments(
        device=device, base_field=SoccerPitchSNCircleCentralSplit()
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

    return cam
