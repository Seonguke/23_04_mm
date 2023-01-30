import time
from collections import OrderedDict
from pathlib import Path

import torch
from lib.structures.field_list import collect

from lib import utils, logger, config, modeling, solver, data

import lib.data.transforms2d as t2d
from lib.config import config
from lib.utils.intrinsics import adjust_intrinsic
from lib.structures import DepthMap

from pathlib import Path
import os
from typing import Dict, Any
import lib.visualize as vis
import json
from lib.visualize.image import write_detection_image, write_depth
from lib.structures.frustum import compute_camera2frustum_transform
import numpy as np
from PIL import Image
import random
from scipy import ndimage as ndi
import copy
import wandb
class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.checkpointer = None
        self.dataloader = None
        self.logger = logger
        self.meters = utils.MetricLogger(delimiter="  ")
        self.checkpoint_arguments = {}

        self.setup()

    def setup(self) -> None:
        # Setup model
        #TODO Change Traning mode
        #self.model = modeling.Masking_pretrain()

        self.model = modeling.PanopticReconstruction()

        device = torch.device(config.MODEL.DEVICE)
        self.model.to(device, non_blocking=True)

        Non_pret = self.model.state_dict()
        update_dict = copy.deepcopy(Non_pret)
        self.model.log_model_info()
        self.model.fix_weights()

        # Setup optimizer, scheduler, checkpointer
        self.optimizer = torch.optim.Adam(self.model.parameters(), config.SOLVER.BASE_LR,
                                          betas=(config.SOLVER.BETA_1, config.SOLVER.BETA_2),
                                          weight_decay=config.SOLVER.WEIGHT_DECAY)
        self.scheduler = solver.WarmupMultiStepLR(self.optimizer, config.SOLVER.STEPS, config.SOLVER.GAMMA,
                                                  warmup_factor=1,
                                                  warmup_iters=0,
                                                  warmup_method="linear")

        output_path = Path(config.OUTPUT_DIR)
        self.checkpointer = utils.DetectronCheckpointer(self.model, self.optimizer, self.scheduler, output_path)

        # Load the checkpoint
        checkpoint_data = self.checkpointer.load()

        # Additionally load a 2D model which overwrites the previously loaded weights
        # TODO: move to checkpointer?

        print(config.MODEL.PRETRAIN2D)
        if config.MODEL.PRETRAIN2D:

            prt_2d = ['encoder2d', 'depth2d', 'instance2d']


            pretrain_2d = torch.load(config.MODEL.PRETRAIN2D)

            for k, v in pretrain_2d["model"].items():
                zz = k.split('.')[0]
                if zz in prt_2d:
                    update_dict[k]=v


            #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
            self.model.load_state_dict(update_dict)

        self.checkpoint_arguments["iteration"] = 0

        if config.SOLVER.LOAD_SCHEDULER:
            self.checkpoint_arguments.update(checkpoint_data)

        # Dataloader
        self.dataloader = data.setup_dataloader(config.DATASETS.TRAIN)

    def do_train(self) -> None:
        # Log start logging

        self.logger.info(f"Start training {self.checkpointer.output_path.name}")

        # Switch training mode
        self.model.switch_training()

        # Main loop
        iteration = 0
        #
        iteration_end = time.time()
        wandb.init(project="PanopticFCN_base", reinit=True)
        for idx, (image_ids, targets) in enumerate(self.dataloader):
            assert targets is not None, "error during data loading"
            data_time = time.time() - iteration_end

            # Get input images
            images = collect(targets, "color")

            # Pass through model
            #losses, results = self.model(images, targets)
            #print(image_ids[0])
            try:
                losses, results = self.model(images, targets)
            except Exception as e:
                print(e, "skipping", image_ids[0])
                del targets, images
                continue

            # Accumulate total loss
            total_loss: torch.Tensor = 0.0
            log_meters = OrderedDict()
            wandb.log(losses)
            for loss_group in losses.values():
                for loss_name, loss in loss_group.items():
                    if torch.is_tensor(loss) and not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss
                        log_meters[loss_name] = loss.item()

            # Loss backpropagation, optimizer & scheduler step
            self.optimizer.zero_grad()

            if torch.is_tensor(total_loss):
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                log_meters["total"] = total_loss.item()
            else:
                log_meters["total"] = total_loss

            # Minkowski Engine recommendation
            torch.cuda.empty_cache()
            if iteration % 1 == 0 and iteration >1000000000000 :
                results["depth"] = [target.get_field("depth") for target in targets]
                input_image = collect(targets, "color")
                #input_image = np.array(input_image)
                # results["rgb"] = collect(targets, "rgb")
                results["input"] = input_image
                output_path = './output/64/' + str(iteration) + '/'

                #self.vis_tr(results,output_path)
                self.vis_64(results, output_path)
                results = {"frustum": {},
                           'panoptic': {}}
                output_path = './output/GT/' + str(iteration) + '/'
                data_time = time.time() - iteration_end
                zzz = targets
                results["input"] = collect(targets, "color")
                im_idx = str(image_ids[-1]).split('/')

                d_root = '/home/sonic/PycharmProjects/front3d/' + str(im_idx[0]) + '/rgb_' + str(im_idx[1]) + '.png'
                zz = 0
                input_image = Image.open(d_root, formats=["PNG"])

                input_image = input_image.convert("RGB")
                input_image = np.array(input_image)
                # results["rgb"] = collect(targets, "rgb")
                results["rgb"] = input_image
                results["depth"] = [target.get_field("depth") for target in targets]
                results["instance2d"] = [target.get_field("instance2d") for target in targets]

                results["frustum"]["geometry"] = collect(targets, "occupancy_256")
                results["panoptic"]["panoptic_instances"] = collect(targets, "instance3d")
                results["panoptic"]["panoptic_semantics"] = collect(targets, "semantic3d")

                self.vis_gt(results, output_path)
            # Save checkpoint
            if iteration % config.SOLVER.CHECKPOINT_PERIOD == 0:
                self.checkpointer.save(f"model_{iteration:07d}", **self.checkpoint_arguments)

            last_training_stage = self.model.set_current_training_stage(iteration)

            # Save additional checkpoint after hierarchy level
            if last_training_stage is not None:
                self.checkpointer.save(f"model_{last_training_stage}_{iteration:07d}", **self.checkpoint_arguments)
                self.logger.info(f"Finish {last_training_stage} hierarchy level")

            # Gather logging information
            self.meters.update(**log_meters)
            batch_time = time.time() - iteration_end
            self.meters.update(time=batch_time, data=data_time)
            current_learning_rate = self.scheduler.get_lr()[0]
            current_training_stage = self.model.get_current_training_stage()

            self.logger.info(self.meters.delimiter.join([f"IT: {iteration:06d}", current_training_stage,
                                                         f"{str(self.meters)}", f"LR: {current_learning_rate}"]))

            iteration += 1
            iteration_end = time.time()

        self.checkpointer.save("model_final", **self.checkpoint_arguments)
    def vis_64(self,results: Dict[str, Any], output_path: os.PathLike) -> None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        depth_map: DepthMap = results["depth"]
        camera2frustum = compute_camera2frustum_transform(depth_map[0].intrinsic_matrix.cpu(),
                                                          torch.tensor(results["input"].size()) / 2.0,
                                                          config.MODEL.PROJECTION.DEPTH_MIN,
                                                          config.MODEL.PROJECTION.DEPTH_MAX,
                                                          config.MODEL.PROJECTION.VOXEL_SIZE)
        camera2frustum[:3, 3] += (torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])) / 2
        frustum2camera = torch.inverse(camera2frustum)
        # print(frustum2camera)
        surface=results["frustum"]["occupancy_64"].squeeze()
        instance=results["frustum"]["instance3d_64"]
        semantic=results["frustum"]["semantic3d_64"]
        vis.write_distance_field(instance.squeeze(), instance.squeeze(), output_path / "mesh_geometry_64_inst.ply", transform=frustum2camera)
        vis.write_distance_field(surface, None, output_path / "mesh_geometry_64.ply", transform=frustum2camera)
        vis.write_distance_field(semantic.squeeze(), semantic.squeeze(), output_path / "mesh_geometry_64_sem.ply",
                                 transform=frustum2camera)
    def vis_tr(self,results: Dict[str, Any], output_path: os.PathLike) -> None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
        min_coordinates = torch.IntTensor([0, 0, 0]).to('cuda')
        debug_check_int_tensor = isinstance(min_coordinates, torch.IntTensor)
        debug_check_int_cuda_tensor = isinstance(min_coordinates, torch.cuda.IntTensor)
        truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE
        depth_map: DepthMap = results["depth"]
        geometry = results["frustum"]["geometry"]
        surface, _, _ = geometry.dense(dense_dimensions, min_coordinates, default_value=truncation)
        instances = results["panoptic"]["panoptic_instances"]
        semantics = results["panoptic"]["panoptic_semantics"]

        # Main outputs
        camera2frustum = compute_camera2frustum_transform(depth_map[0].intrinsic_matrix.cpu(),
                                                          torch.tensor(results["input"].size()) / 2.0,
                                                          config.MODEL.PROJECTION.DEPTH_MIN,
                                                          config.MODEL.PROJECTION.DEPTH_MAX,
                                                          config.MODEL.PROJECTION.VOXEL_SIZE)

        # remove padding: original grid size: [256, 256, 256] -> [231, 174, 187]
        camera2frustum[:3, 3] += (torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])) / 2
        frustum2camera = torch.inverse(camera2frustum)
        # print(frustum2camera)
        vis.write_distance_field(surface.squeeze(), None, output_path / "mesh_geometry.ply", transform=frustum2camera)
        vis.write_distance_field(surface.squeeze(), instances.squeeze(), output_path / "mesh_instances.ply",
                                 transform=frustum2camera)
        vis.write_distance_field(surface.squeeze(), semantics.squeeze(), output_path / "mesh_semantics.ply",
                                 transform=frustum2camera)
    def vis_gt(self,results: Dict[str, Any], output_path: os.PathLike) -> None:
        device = results["input"].device
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        from torchvision.transforms.functional import to_pil_image
        import matplotlib.pyplot as plt
        seg_map = results["instance2d"][0]
        # plt.figure()
        # plt.subplot(1, 2, 1)



        write_detection_image(results["rgb"], seg_map, output_path / "seg_map.png")

        #    print(output_path)
        # Visualize depth prediction
        depth_map: DepthMap = results["depth"][0]
        depth_map.to_pointcloud(output_path / "GT_depth_GT.ply")
        write_depth(depth_map, output_path / "GT_depth_map.png")

        # Visualize 2D detections
        # write_detection_image(results["input"], results["instance"], output_path / "detection.png")

        # Visualize projection
        #vis.write_pointcloud(results["projection"].C[:, 1:], None, output_path / "projection.ply")

        # Visualize 3D outputs
        dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
        min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
        truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE

        surface = results["frustum"]["geometry"]
        #surface, _, _ = geometry.dense(dense_dimensions, min_coordinates, default_value=truncation)
        instances = results["panoptic"]["panoptic_instances"]
        semantics = results["panoptic"]["panoptic_semantics"]

        # Main outputs
        camera2frustum = compute_camera2frustum_transform(depth_map.intrinsic_matrix.cpu(),
                                                          torch.tensor(results["input"].size()) / 2.0,
                                                          config.MODEL.PROJECTION.DEPTH_MIN,
                                                          config.MODEL.PROJECTION.DEPTH_MAX,
                                                          config.MODEL.PROJECTION.VOXEL_SIZE)

        # remove padding: original grid size: [256, 256, 256] -> [231, 174, 187]
        camera2frustum[:3, 3] += (torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])) / 2
        frustum2camera = torch.inverse(camera2frustum)

        vis.write_distance_field(surface.squeeze(), None, output_path / "mesh_geometry.ply", transform=frustum2camera)
        vis.write_distance_field( surface.squeeze(),instances.squeeze(), output_path / "mesh_instances.ply",
                                 transform=frustum2camera)
        vis.write_distance_field(surface.squeeze(), semantics.squeeze(), output_path / "mesh_semantics.ply",
                                 transform=frustum2camera)
        # with open(output_path / "semantic_classes.json", "w") as f:
        #      json.dump(results["panoptic"]["panoptic_semantic_mapping"], f, indent=4)
        #
        # # Visualize auxiliary outputs
        # #vis.write_pointcloud(geometry.C[:, 1:], None, output_path / "sparse_coordinates.ply")
        # surface[surface<3]=3
        # surface_mask = surface.squeeze() < iso_value
        # points = surface_mask.squeeze().nonzero()
        # point_semantics = semantics.squeeze()[surface_mask]
        # point_instances = instances.squeeze()[surface_mask]
        #
        # vis.write_pointcloud(points, None, output_path / "points_geometry.ply")
        # vis.write_semantic_pointcloud(points, point_semantics, output_path / "points_surface_semantics.ply")
        # vis.write_semantic_pointcloud(points, point_instances, output_path / "points_surface_instances.ply")
    def masking_Occupancy(self,results,per : int):


        #0,1 ,11,12
        #TODO Add Occupancy
        occ = results["frustum"]["geometry"]
        occ_shape= occ.shape
        i_shape= results["panoptic"]["panoptic_instances"].shape
        occ=occ.view(-1)

        instance = results["panoptic"]["panoptic_instances"].cpu()
        # make a little 3D diamond:
        #TODO Dilated Mask############################################
        mask3d =instance>0
        mask3d=mask3d.reshape((256,256,256)).cpu().numpy()
        diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
        # dilate 30x with it
        dilated = ndi.binary_dilation(np.array(mask3d), diamond, iterations=3)
        # 0,1 ,11,12
        dilated=~dilated
        dilated=torch.from_numpy(dilated).to('cuda')
        dilated=dilated.view(-1)
        ###########################################################3

        instance = instance.view(-1)
        #= torch.mul(occ,wall_mask.to('cuda'))
        un=np.unique(np.array(instance))
        for i in un:
            if i==0 :
                continue
            cnt= instance==i
            cnt=cnt.sum().item()
            idx= np.where(instance ==i)
            sam=random.sample(list(idx[0]),int(cnt*(per/100)))
            instance[sam]=0 #instance to Occupancy
        v_mask = (instance>0)
        v_mask = (v_mask).to('cuda')
        v_mask = torch.logical_or(v_mask, dilated)
        occ= torch.mul(occ,v_mask)

        #np.multiply(v_mask,instance)
        occ=occ.reshape(occ_shape)

        results["frustum"]["geometry"]=occ
        return results

    def GT_vis(self):

        self.logger.info(f"Start training hi")

        # Switch training mode
        self.model.switch_training()

        # Main loop
        iteration = 0
        iteration_end = time.time()

        for idx, (image_ids, targets) in enumerate(self.dataloader):
            assert targets is not None, "error during data loading"
            results = {"frustum":{},
                       'panoptic':{}}
            output_path='./output/GT/'+str(idx)+'/'
            data_time = time.time() - iteration_end
            zzz= targets
            results["input"]=collect(targets, "color")
            im_idx= str(image_ids[-1]).split('/')

            d_root='/home/sonic/PycharmProjects/front3d/'+str(im_idx[0])+'/rgb_'+str(im_idx[1])+'.png'
            zz=0
            input_image =Image.open(d_root, formats=["PNG"])

            input_image = input_image.convert("RGB")
            input_image = np.array(input_image)
            #results["rgb"] = collect(targets, "rgb")
            results["rgb"]=input_image
            results["depth"] = [target.get_field("depth") for target in targets]
            results["instance2d"] = [target.get_field("instance2d") for target in targets]

            results["frustum"]["geometry"] = collect(targets, "occupancy_256")
            results["panoptic"]["panoptic_instances"]= collect(targets, "instance3d")
            results["panoptic"]["panoptic_semantics"]= collect(targets, "semantic3d")

            #output_path = Path(output_path)
            self.masking_Occupancy(results,90)#2: 50%
            self.vis_gt(results,output_path)




