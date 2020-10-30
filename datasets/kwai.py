import pickle as p
from copy import deepcopy
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch

from cosypose.utils.logging import get_logger
logger = get_logger(__name__)

def ply_vtx(path):
    """
    discription: read all vertices from a ply file
    (borrow from https://github.com/paroj/linemod_dataset/blob/master/read.py)

    : param path: path to ply model file
    return: all 3D points of ply model: (N, 3)
    """
    with open(path) as f:
        assert f.readline().strip() == "ply"
        while True:
            line = f.readline().strip()
            if line.startswith('element vertex'):
                N = int(line.split()[-1])
            if line.startswith('element face'):
                F = int(line.split()[-1])
            if line == 'end_header':
                break
        pts = []
        for _ in range(N):
            pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)


class KwaiCamInfo:
    fx = 2100.
    fy = 2100.
    cx = 540.
    cy = 960.
    K = np.asarray([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.] 
        ])
    T0C = np.identity(4)
    TC0 = T0C
    
# kwai single frame dataset
train_test_ratio = 0.15
class KwaiDataset:
    def __init__(self, ds_dir, split="train"):
        # init path
        ds_dir = Path(ds_dir)
        self.ds_dir = ds_dir
        assert ds_dir.exists(), 'Dataset does not exists.'
        self.rgb_dir = ds_dir / "frames"
        self.gt_dir = ds_dir / "poses"
        self.coor_dir = ds_dir / "coor"
        self.rendered_depth_dir = ds_dir / "render_dir"

        # load split txt
        self.split = split
        self.split_idx_list = np.loadtxt(
                fname=str(ds_dir / "split" / f"{split}_{train_test_ratio:.2f}.txt"),
                dtype=np.int, 
                delimiter='\n', 
            )
        self.len = len(self.split_idx_list)
        


    def __len__(self):
        return self.len

    def __getitem__(self, frame_id):
        # rgb
        rgb_path = self.rgb_dir / f"{frame_id:06d}.jpg"
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        rgb = torch.as_tensor(rgb)

        # mask 
        depth_path = self.rendered_depth_dir / f"{frame_id:06d}.png"
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask = (depth > 0).astype(np.uint8) # 0-1
        mask = torch.as_tensor(mask)

        # camera info 
        camera = dict(
            T0C=deepcopy(KwaiCamInfo.T0C), 
            K=deepcopy(KwaiCamInfo.K),
            TWC=deepcopy(KwaiCamInfo.T0C),
            resolution=rgb.shape[:2]
        )

        # objects info
        objects = []
        
        RtCO = np.loadtxt(
            fname=self.gt_dir / f"frame{frame_id + 1:04d}.txt"
        )
        TCO = np.identity(4)
        TCO[:3, :3] = deepcopy(RtCO)
        T0O = KwaiCamInfo.T0C @ TCO

        obj_id = 1
        # coor
        coor_path = self.coor_dir / f"{frame_id:06d}.pkl"
        with open(coor_path, "rb") as f:
            coor_info = p.load(f)
        
        x = coor_path["u"]
        y = coor_path["v"]
        w = coor_path["w"]
        h = coor_path["h"]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        obj = dict(
            label=obj_id, 
            name=obj_id, 
            TWO=T0O, 
            T0O=T0O,
            visib_fract=-1, 
            id_in_segm=1, 
            bbox=[x1, y1, x2, y2] 
        )
        del coor_info
        
        # whole meta
        obs = dict(
            objects=objects,
            camera=camera, 
            frame_info=dict(
                frame_id=frame_id
            )
        )        
        
        return rgb, mask, obs

class KwaiTrackingDataset:
    def __init__(self, ds_dir, split="train"):
        # init path
        self.base_ds = KwaiDataset(ds_dir, split)
        self.len = len(self.base_ds) - 1 # minus 1 means the length of pairs
    

    def __len__(self):
        return self.len

    def __getitem__(self, frame_id):
        # return a pair of objects
        src_rgb, src_mask, src_obs = self.base_ds[frame_id]
        dest_rgb, dest_mask, dest_obs = self.base_ds[frame_id + 1]
        return src_rgb, src_mask, src_obs, \ 
            dest_rgb, dest_mask, dest_obs

class KwaiObjectDataset:
    def __init__(self, ds_dir):
        ds_dir = Path(ds_dir)
        self.model_eval_path = ds_dir / "yuanqisenlin_transformed_eval.ply"
        self.model_mesh_path = ds_dir / "yuanqisenlin_transformed.ply"
        pcd = ply_vtx(str(self.model_eval_path))
        # unit: mm !!! should be change to m later
        # pcd *= 1e3 # from m to mm
        
        min_x, min_y, min_z = np.min(pcd, axis=0)
        max_x, max_y, max_z = np.max(pcd, axis=0)
        models_info = {
            1: {
                "max_x": max(max_x, abs(min_x)), 
                "max_y": max(max_y, abs(min_y)), 
                "max_z": max(max_z, abs(min_z)), 
                "min_x": min_x, 
                "min_y": min_y, 
                "min_z": min_z, 
                "size_x": max_x - min_x, 
                "size_y": max_y - min_y, 
                "size_z": max_z - min_z, 
                "diameter": np.sqrt(
                    (max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2
                    )
            }
        }


        objects = []
        obj = dict(
            label=1, 
            category=None,
            mesh_path=self.model_mesh_path, 
            mesh_units="m", 
            is_symmetric=False,
            scale=1.0,
            diameter=models_info["diameter"], 
            diameter_m=models_info["diameter"], 

        )
        objects.append(obj)

        self.objects = objects
        self.ds_dir = ds_dir

    def __getitem__(self, idx):
        return self.objects[idx]

    def __len__(self):
        return len(self.objects)