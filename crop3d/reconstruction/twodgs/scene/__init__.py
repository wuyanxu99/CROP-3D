#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import (
    sceneLoadTypeCallbacks,
    storePly,
    getNerfppNorm,
    SceneInfo,
)
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.trajectory_axis_utils import (
    apply_align_to_camera_infos,
    apply_align_to_pcd,
    build_align_to_world_z,
    compute_trajectory_axis_and_up,
    compute_trajectory_segment,
    filter_pcd_by_axis_depth,
)

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.trajectory_axis_enable = False
        self.trajectory_depth_min = 0.0
        self.trajectory_depth_max = 0.0

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            traj_on = getattr(args, "trajectory_axis_enable", False)
            preserve_world = bool(getattr(args, "trajectory_preserve_world_coords", False))
            pcd_src = scene_info.point_cloud
            if traj_on and pcd_src is not None:
                train_cams = list(scene_info.train_cameras)
                test_cams = list(scene_info.test_cameras)
                pcd = pcd_src
                K = int(getattr(args, "trajectory_cameras_per_frame", 1))
                buf = float(getattr(args, "trajectory_segment_buffer", 0.5))
                dmax = float(getattr(args, "trajectory_depth_max", 0.0))
                dmin = float(getattr(args, "trajectory_depth_min", 0.0))

                if dmax > 0:
                    o0, ax0, tm0, tM0 = compute_trajectory_segment(train_cams, K, buf)
                    pcd = filter_pcd_by_axis_depth(pcd, o0, ax0, dmin, dmax, tm0, tM0)

                if not preserve_world:
                    all_cams = train_cams + test_cams
                    traj_ax, up_avg, origin_pca = compute_trajectory_axis_and_up(all_cams)
                    R_align, t_align = build_align_to_world_z(traj_ax, up_avg, origin_pca)
                    pcd = apply_align_to_pcd(pcd, R_align, t_align)
                    train_cams = apply_align_to_camera_infos(train_cams, R_align, origin_pca)
                    test_cams = apply_align_to_camera_infos(test_cams, R_align, origin_pca)

                nerf_norm = getNerfppNorm(train_cams)
                scene_info = SceneInfo(
                    point_cloud=pcd,
                    train_cameras=train_cams,
                    test_cameras=test_cams,
                    nerf_normalization=nerf_norm,
                    ply_path=scene_info.ply_path,
                )

                to, ta, ttmin, ttmax = compute_trajectory_segment(train_cams, K, buf)
                self.trajectory_axis_enable = True
                self.trajectory_depth_min = dmin
                self.trajectory_depth_max = dmax
                self.trajectory_origin = to.astype(np.float32)
                self.trajectory_axis = ta.astype(np.float32)
                self.trajectory_t_min = float(ttmin)
                self.trajectory_t_max = float(ttmax)

                inp = os.path.join(self.model_path, "input.ply")
                rgb = (np.clip(np.asarray(pcd.colors), 0.0, 1.0) * 255.0).astype(np.uint8)
                storePly(inp, np.asarray(pcd.points), rgb)

                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                    json.dump(json_cams, file)
            else:
                if traj_on and pcd_src is None:
                    print(
                        "[Warning] trajectory_axis_enable set but point cloud missing; "
                        "skipping trajectory preprocessing."
                    )
                with open(scene_info.ply_path, "rb") as src_file, open(
                    os.path.join(self.model_path, "input.ply"), "wb"
                ) as dest_file:
                    dest_file.write(src_file.read())
                json_cams = []
                camlist = []
                if scene_info.test_cameras:
                    camlist.extend(scene_info.test_cameras)
                if scene_info.train_cameras:
                    camlist.extend(scene_info.train_cameras)
                for id, cam in enumerate(camlist):
                    json_cams.append(camera_to_JSON(id, cam))
                with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                    json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
