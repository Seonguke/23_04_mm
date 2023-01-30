import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from typing import Dict, Any

from lib import modeling
import matplotlib.pyplot as plt
import lib.data.transforms2d as t2d
from lib.config import config
from lib.utils.intrinsics import adjust_intrinsic
from lib.structures import DepthMap

import lib.visualize as vis
from lib.visualize.image import write_detection_image, write_depth
from lib.structures.frustum import compute_camera2frustum_transform

import argparse
import os
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import Tuple, Dict

from lib import modeling, metrics, visualize

from lib.data import setup_dataloader

from lib.modeling.utils import thicken_grid
from lib.visualize.mesh import get_mesh

from lib.config import config
from lib.structures.field_list import collect


from tools.test_net_single_image import configure_inference
from lib.utils import re_seed
from lib.metrics.panoptic_quality import PQStatCategory
import os

def dishow(disp):
    plt.imshow(disp)
    plt.jet()
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth2Disparity image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.plot
    plt.show()


def main(opts):
    configure_inference(opts)

    device = torch.device("cuda:0")

    # Define model and load checkpoint.
    print("Load model...")
    model = modeling.PanopticReconstruction()
    checkpoint = torch.load(opts.model)
    model.load_state_dict(checkpoint["model"])  # load model checkpoint
    model = model.to(device)  # move to gpu
    model.switch_test()

    # Define image transformation.
    color_image_size = (320, 240)
    depth_image_size = (160, 120)

    imagenet_stats = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image_transforms = t2d.Compose([
        t2d.Resize(color_image_size),
        t2d.ToTensor(),
        t2d.Normalize(imagenet_stats[0], imagenet_stats[1]),  # use imagenet stats to normalize image
    ])

    # Open and prepare input image.
    print("Load input image...")
    input_image = Image.open(opts.input)
    input_image = input_image.convert("RGB")
    dishow(np.asarray(input_image))
    input_image = image_transforms(input_image)
    input_image = input_image.unsqueeze(0).to(device)

    # Prepare intrinsic matrix.
    front3d_intrinsic = np.array(config.MODEL.PROJECTION.INTRINSIC)
    front3d_intrinsic = adjust_intrinsic(front3d_intrinsic, color_image_size, depth_image_size)
    front3d_intrinsic = torch.from_numpy(front3d_intrinsic).to(device).float()

    # Prepare frustum mask.
    front3d_frustum_mask = np.load(str("data/frustum_mask.npz"))["mask"]
    front3d_frustum_mask = torch.from_numpy(front3d_frustum_mask).bool().to(device).unsqueeze(0).unsqueeze(0)

    print("Perform panoptic 3D scene reconstruction...")
    with torch.no_grad():
        results = model.inference(input_image, front3d_intrinsic, front3d_frustum_mask)

    print(f"Visualize results, save them at {config.OUTPUT_DIR}")
    visualize_results(results, config.OUTPUT_DIR)
    visualize_results2(results, config.OUTPUT_DIR)



def configure_inference(opts):
    # load config
    config.OUTPUT_DIR = opts.output
    config.merge_from_file(opts.config_file)
    config.merge_from_list(opts.opts)
    # inference settings
    config.MODEL.FRUSTUM3D.IS_LEVEL_64 = False
    config.MODEL.FRUSTUM3D.IS_LEVEL_128 = False
    config.MODEL.FRUSTUM3D.IS_LEVEL_256 = False
    config.MODEL.FRUSTUM3D.FIX = True
def visualize_results2(results: Dict[str, Any], output_path: os.PathLike) -> None:
    device = results["input"].device
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Visualize depth prediction
    depth_map: DepthMap = results["depth"]
    depth_map.to_pointcloud(output_path / "depth_prediction.ply")
    write_depth(depth_map, output_path / "depth_map.png")

    # Visualize 2D detections
    # write_detection_image(results["input"], results["instance"], output_path / "detection.png")

    # Visualize projection
    vis.write_pointcloud(results["projection"].C[:, 1:], None, output_path / "projection.ply")

    # Visualize 3D outputs
    dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
    min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
    truncation = config.MODEL.FRUSTUM3D.TRUNCATION
    iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE

    geometry = results["frustum"]["geometry"]
    surface, _, _ = geometry.dense(dense_dimensions, min_coordinates, default_value=truncation)
    instances = results["panoptic"]["panoptic_instances"]
    semantics = results["panoptic"]["panoptic_semantics"]

    # Main outputs
    camera2frustum = compute_camera2frustum_transform(depth_map.intrinsic_matrix.cpu(), torch.tensor(results["input"].size()) / 2.0,
                                                      config.MODEL.PROJECTION.DEPTH_MIN,
                                                      config.MODEL.PROJECTION.DEPTH_MAX,
                                                      config.MODEL.PROJECTION.VOXEL_SIZE)


    # remove padding: original grid size: [256, 256, 256] -> [231, 174, 187]
    camera2frustum[:3, 3] += (torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])) / 2
    frustum2camera = torch.inverse(camera2frustum)
    #print(frustum2camera)
    vis.write_distance_field(surface.squeeze(), None, output_path / "mesh_geometry.ply", transform=frustum2camera)
    vis.write_distance_field(surface.squeeze(), instances.squeeze(), output_path / "mesh_instances.ply", transform=frustum2camera)
    vis.write_distance_field(surface.squeeze(), semantics.squeeze(), output_path / "mesh_semantics.ply", transform=frustum2camera)

    with open(output_path / "semantic_classes.json", "w") as f:
        json.dump(results["panoptic"]["panoptic_semantic_mapping"], f, indent=4)

    # Visualize auxiliary outputs
    vis.write_pointcloud(geometry.C[:, 1:], None, output_path / "sparse_coordinates.ply")

    surface_mask = surface.squeeze() < iso_value
    points = surface_mask.squeeze().nonzero()
    point_semantics = semantics[surface_mask]
    point_instances = instances[surface_mask]

    vis.write_pointcloud(points, None, output_path / "points_geometry.ply")
    vis.write_semantic_pointcloud(points, point_semantics, output_path / "points_surface_semantics.ply")
    vis.write_semantic_pointcloud(points, point_instances, output_path / "points_surface_instances.ply")

def visualize_results(results: Dict[str, Any], output_path: os.PathLike) -> None:
    device = results["input"].device
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Visualize depth prediction
    depth_map: DepthMap = results["depth"]
    #depth_map.to_pointcloud(output_path / "GT_depth_GT.ply")
    depth_map.to_pointcloud(output_path / "cam_depth_prediction.ply")
    zzz = depth_map
    zzz = zzz.depth_map
    write_depth(zzz, output_path / "depth_map.png")

    # Visualize projection
    vis.write_pointcloud(results["projection"].C[:, 1:], None, output_path / "projection.ply")

    # Visualize 3D outputs
    dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
    min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
    truncation = config.MODEL.FRUSTUM3D.TRUNCATION
    iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE

    geometry = results["frustum"]["geometry"]
    surface, _, _ = geometry.dense(dense_dimensions, min_coordinates, default_value=truncation)

    # Main outputs
    camera2frustum = compute_camera2frustum_transform(depth_map.intrinsic_matrix.cpu(),
                                                      torch.tensor(results["input"].size()) / 2.0,
                                                      config.MODEL.PROJECTION.DEPTH_MIN,
                                                      config.MODEL.PROJECTION.DEPTH_MAX,
                                                      config.MODEL.PROJECTION.VOXEL_SIZE)

    # remove padding: original grid size: [256, 256, 256] -> [231, 174, 187]
    camera2frustum[:3, 3] += (torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])) / 2

    frustum2camera = torch.inverse(camera2frustum)
    # print(frustum2camera)
    vis.write_distance_field(surface.squeeze(), None, output_path / "mesh_geometry.ply", transform=frustum2camera)

    # Visualize auxiliary outputs
    vis.write_pointcloud(geometry.C[:, 1:], None, output_path / "sparse_coordinates.ply")

    surface_mask = surface.squeeze() < iso_value
    points = surface_mask.squeeze().nonzero()

    vis.write_pointcloud(points, None, output_path / "points_geometry.ply")
    #depth_map.to_pointcloud(output_path / "Frust_depth_prediction.ply", camera2frustum=camera2frustum)

def vis_gt( results: Dict[str, Any], output_path: os.PathLike) -> None:
    device = results["input"].device
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Visualize depth prediction
    depth_map: DepthMap = results["depth"][0]
    zzz= np.array(depth_map.depth_map.cpu())
    zzz=np.flip(zzz,axis=0)
    zzz=np.flip(zzz,axis=1)
    dishow(zzz)
    depth_map.depth_map = torch.tensor(zzz.copy())
    depth_map.to_pointcloud(output_path / "cam_GT_depth_GT.ply") # Camera space
    write_depth(depth_map.depth_map, output_path / "GT_depth_map.png")

    # Visualize 3D outputs
    dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
    min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
    truncation = config.MODEL.FRUSTUM3D.TRUNCATION
    iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE

    surface = results["frustum"]["geometry"]
    # surface, _, _ = geometry.dense(dense_dimensions, min_coordinates, default_value=truncation)
    # Main outputs
    camera2frustum = compute_camera2frustum_transform(depth_map.intrinsic_matrix.cpu(),
                                                      torch.tensor(results["input"].size()) / 2.0,
                                                      config.MODEL.PROJECTION.DEPTH_MIN,
                                                      config.MODEL.PROJECTION.DEPTH_MAX,
                                                      config.MODEL.PROJECTION.VOXEL_SIZE)

    # remove padding: original grid size: [256, 256, 256] -> [231, 174, 187]
    camera2frustum[:3, 3] += (torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])) / 2
    frustum2camera = torch.inverse(camera2frustum)

    vis.write_distance_field(surface.squeeze(), None, output_path / "cam_mesh_geometry.ply", transform=frustum2camera)

    vis.write_distance_field(surface.squeeze(), None, output_path / "frust_mesh_geometry.ply")

    surface_mask = surface.squeeze() < iso_value
    points = surface_mask.squeeze().nonzero()

    vis.write_pointcloud(points, None, output_path / "points_geometry.ply")
    #depth_map.to_pointcloud(output_path / "GT_Frust_depth_prediction.ply", camera2frustum=camera2frustum)

if __name__ == '__main__':

    pth = '/home/sonic/PycharmProjects/front3d/'
    f = open("resources/front3d/test_list_3d.txt", 'r')
    lines = f.readlines()
    cnt = 0
    config.DATASETS.TEST = "Front3D_Test"
    dataloader = setup_dataloader(config.DATASETS.TEST, False, is_iteration_based=False, shuffle=False)
    dataloader.dataset.samples = dataloader.dataset.samples[0:100]
    for idx, (image_ids, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if targets is None:
            print(f"Error, {image_ids[0]}")
            continue

        # Get input images
        images = collect(targets, "color")

        im_idx = str(image_ids[-1]).split('/')

        d_root = '/home/sonic/PycharmProjects/front3d/' + str(im_idx[0]) + '/rgb_' + str(im_idx[1]) + '.png'

        parser = argparse.ArgumentParser()
        parser.add_argument("--input", "-i", type=str, default=str(d_root))
        parser.add_argument("--output", "-o", type=str, default="output/projection_test/" + str(cnt) + '/')
        parser.add_argument("--config-file", "-c", type=str, default="configs/front3d_sample.yaml")
        parser.add_argument("--model", "-m", type=str, default="data/panoptic_front3d_v2.pth")

        parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

        args = parser.parse_args()
        main(args)

        results = {"frustum": {},
                   'panoptic': {}}
        output_path = './output/GT/' + str(cnt) + '/'
        results["input"] = collect(targets, "color")

        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        input_image = Image.open(d_root, formats=["PNG"])
        input_image.save( './output/GT/' + str(cnt) + '/'+'rgb.png')
        input_image = input_image.convert("RGB")
        input_image = np.array(input_image)
        # results["rgb"] = collect(targets, "rgb")
        results["rgb"] = images
        results["depth"] = [target.get_field("depth") for target in targets]

        results["frustum"]["geometry"] = collect(targets, "geometry")
        vis_gt(results,output_path)
        cnt = cnt + 1

