import os.path as osp
import time
import cv2
import torch
import numpy as np
from src.utils.data_utils import get_K_crop_resize, get_image_crop_resize
from src.utils.vis_utils import reproj

class YoloV5Detector():
    def __init__(self, yolov5_pth, weights_pth):
        self.model = torch.hub.load(yolov5_pth, 'custom', path=weights_pth, source='local')

    def crop_img_by_bbox(self, origin_img, bbox, K=None, crop_size=512):
        """
        Crop image by detect bbox
        Input:
            query_img_path: str,
            bbox: np.ndarray[x0, y0, x1, y1],
            K[optional]: 3*3
        Output:
            image_crop: np.ndarray[crop_size * crop_size],
            K_crop[optional]: 3*3
        """
        x0, y0 = bbox[0], bbox[1]
        x1, y1 = bbox[2], bbox[3]

        resize_shape = np.array([y1 - y0, x1 - x0])
        if K is not None:
            K_crop, K_crop_homo = get_K_crop_resize(bbox, K, resize_shape)
        image_crop, trans1 = get_image_crop_resize(origin_img, bbox, resize_shape)
        
        '''
        bbox_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape = np.array([crop_size, crop_size])
        if K is not None:
            K_crop, K_crop_homo = get_K_crop_resize(bbox_new, K_crop, resize_shape)
        image_crop, trans2 = get_image_crop_resize(image_crop, bbox_new, resize_shape)
        '''
        
        return image_crop, K_crop if K is not None else None

    def detect(self, query_img, K, crop_size=512):
        """
        Detect object by YoloV5 and crop image.
        Input:
            query_image: np.ndarray[1*1*H*W],
            query_img_path: str,
            K: np.ndarray[3*3], intrinsic matrix of original image
        Output:
            bounding_box: np.ndarray[x0, y0, x1, y1]
            cropped_image: torch.tensor[1 * 1 * crop_size * crop_size] (normalized),
            cropped_K: np.ndarray[3*3];
        """
        # ensure input is cuda tensor with shape (1,1,h,w)
        #query_inp = torch.from_numpy(query_img)[None][None].cuda()

        # pass through yolo and obtain bounding box
        out_crop = self.model(query_img).crop(save=False)
        if out_crop == []:
        	bbox = np.array([0, 0, query_img.shape[0], query_img.shape[1]])
        	image_crop = query_img
        	K_crop = K
        else:
        	bbox = out_crop[0]['box']
        	x0, y0, x1, y1 = tuple(map(int,bbox))
        	bbox = np.array([x0, y0, x1, y1])
        	image_crop, K_crop = self.crop_img_by_bbox(query_img, bbox, K, crop_size=crop_size)

        return bbox, image_crop, K_crop
    
    def previous_pose_detect(self, query_img, K, pre_pose, bbox3D_corner, crop_size=512):
        """
        Detect object by projecting 3D bbox with estimated last frame pose.
        Input:
            query_image_path: str,
            K: np.ndarray[3*3], intrinsic matrix of original image
            pre_pose: np.ndarray[3*4] or [4*4], pose of last frame
            bbox3D_corner: np.ndarray[8*3], corner coordinate of annotated 3D bbox
        Output:
            bounding_box: np.ndarray[x0, y0, x1, y1]
            cropped_image: torch.tensor[1 * 1 * crop_size * crop_size] (normalized),
            cropped_K: np.ndarray[3*3];
        """
        # Project 3D bbox:
        proj_2D_coor = reproj(K, pre_pose, bbox3D_corner)
        x0, y0 = np.min(proj_2D_coor, axis=0)
        x1, y1 = np.max(proj_2D_coor, axis=0)
        bbox = np.array([x0, y0, x1, y1]).astype(np.int32)

        image_crop, K_crop = self.crop_img_by_bbox(query_img, bbox, K, crop_size=crop_size)

        return bbox, image_crop, K_crop
