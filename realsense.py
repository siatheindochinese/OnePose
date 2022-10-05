import cv2
import torch
import hydra
from tqdm import tqdm
import os.path as osp
import numpy as np
import pyrealsense2 as rs

from torch.utils.data import DataLoader
from src.utils import data_utils, path_utils, eval_utils, vis_utils
from src.yolov5_detector import YoloV5Detector

from inference import load_model, pack_data

def draw_keypoints(img, corners, K_full, K_crop, color=(0, 255, 0), radius=3):
	backtorgb = img.copy()
	translation = K_full @ np.linalg.inv(K_crop)
	for c in np.array(corners):
		c = translation @ np.append(c,1)
		c = c.astype(int)[:2]
		cv2.circle(backtorgb, tuple(c), radius, color, thickness=-1)
	return backtorgb
	
def realtime_reproj(frame, poses, bbox3d, K_full, colors=['g']):
	image_full = frame
	for pose, color in zip(poses, colors):
		# Draw pred 3d box
		if pose is not None:
			reproj_box_2d = vis_utils.reproj(K_full, pose, bbox3d)
			vis_utils.draw_3d_box(image_full, reproj_box_2d, color=color)

	return image_full
	
@torch.no_grad()
@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg):
	# initialize realsense and opencv configurations
	pipe = rs.pipeline()
	#rscfg = rs.config()
	colorizer = rs.colorizer()
	profile = pipe.start()
	align = rs.align(rs.stream.color)
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	width, height = 1280, 720

	# initialize video recorder object
	#writer = cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))

	# load SuperPoint, SuperGlue and OnePose GAT
	matching_model, extractor_model = load_model(cfg)
	
	# load yolov5 detector
	yolov5_detector = YoloV5Detector(cfg.yolov5_dir, cfg.yolov5_weights_dir)

	# load SfM
	anno_dir = osp.join(cfg.sfm_model_dir, f'outputs_{cfg.network.detection}_{cfg.network.matching}', 'anno')
	avg_anno_3d_path = osp.join(anno_dir, 'anno_3d_average.npz')
	clt_anno_3d_path = osp.join(anno_dir, 'anno_3d_collect.npz')
	idxs_path = osp.join(anno_dir, 'idxs.npy')

	num_leaf = cfg.num_leaf
	avg_data = np.load(avg_anno_3d_path)
	clt_data = np.load(clt_anno_3d_path)
	idxs = np.load(idxs_path)
	bbox3d = np.loadtxt(cfg.box3d_path)

	keypoints3d = torch.Tensor(clt_data['keypoints3d']).cuda()
	num_3d = keypoints3d.shape[0]
    ##### Load average 3D features:
	avg_descriptors3d, _ = data_utils.pad_features3d_random(avg_data['descriptors3d'], avg_data['scores3d'], num_3d)

    ##### Load corresponding 2D features of each 3D point:
	clt_descriptors, _ = data_utils.build_features3d_leaves(clt_data['descriptors3d'], clt_data['scores3d'], idxs, num_3d, num_leaf)

	# load intrinsics
	intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
	K_full = np.array([[intrin.fx, 0, intrin.ppx],
				  [0, intrin.fy, intrin.ppy],
				  [0,         0,          1]])
	K_crop = K_full
	
	# abstract object detection pipeline
	def detect_object(inp, init):
		if init == False:
			print('initial object-detection frame')
			bbox, inp_crop, K_crop = yolov5_detector.detect(inp, K_full, crop_size=512)
			init = True
		else:
			if len(previous_inliers) < 8:
				print('object-detection frame')
				bbox, inp_crop, K_crop = yolov5_detector.detect(inp, K_full, crop_size=512)
			else:
				print('GT frame')
				bbox, inp_crop, K_crop = yolov5_detector.previous_pose_detect(inp, K_full, previous_frame_pose, bbox3d, crop_size=512)
		print('bbox=',bbox)
		##### Determine if object is detected within the frame
		object_detected = not (bbox == np.array([0, 0, height, width])).all() #hardcoded dimensions
		print('obj_detected =',object_detected)
		return object_detected, bbox, inp_crop, K_crop, init
		
	# abstract OnePose pipeline
	def onePoseForwardPass(inp_crop, K_crop):
		inp_crop_cuda = torch.from_numpy(inp_crop.astype(np.float32)[None][None]/255.).cuda()
		pred_detection = extractor_model(inp_crop_cuda)
		pred_detection = {k: v[0].detach().cpu().numpy() for k, v in pred_detection.items()}
		inp_data = pack_data(avg_descriptors3d, clt_descriptors, keypoints3d, pred_detection, np.array([512, 512]))
		pred, _ = matching_model(inp_data)
		matches = pred['matches0'].detach().cpu().numpy()
		valid = matches > -1
		notvalid = matches <= -1
		kpts2d = pred_detection['keypoints']
		kpts3d = inp_data['keypoints3d'][0].detach().cpu().numpy()
		confidence = pred['matching_scores0'].detach().cpu().numpy()
		mkpts2d, mkpts3d, mconf = kpts2d[valid], kpts3d[matches[valid]], confidence[valid]
		validcorners = mkpts2d
		notvalidcorners = kpts2d[notvalid]
		print('    ',str(len(validcorners)),'valid keypoints detected')
		_, pose_pred_homo, inliers = eval_utils.ransac_PnP(K_crop, mkpts2d, mkpts3d, scale=1000)

		return pose_pred_homo, inliers, validcorners, notvalidcorners
		
	# Some while-loop flags
	init = False
	previous_frame_pose = np.eye(4)
	previous_inliers = []
	while True:
		# stream the next frame
		frameset = pipe.wait_for_frames()
		frameset = align.process(frameset)
		color_frame = frameset.get_color_frame()
		depth_frame = frameset.get_depth_frame()
		dpt = np.asanyarray(colorizer.colorize(depth_frame).get_data())
		bgr = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)
		frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

		##### object detection
		object_detected, bbox, inp_crop, K_crop, init = detect_object(frame, init)
		
		if object_detected:
			##### process the frame
			print(inp_crop.shape)
			pose_pred_homo, inliers, validcorners, notvalidcorners = onePoseForwardPass(inp_crop, K_crop)
			previous_frame_pose = pose_pred_homo
			previous_inliers = inliers
		else:
			##### Reset stored poses
			previous_frame_pose = np.eye(4)
			previous_inliers = []

			##### Project BBox onto frame (if object detected, else just plot out SPP keypoints)
		if object_detected and not np.array_equal(pose_pred_homo, np.eye(4)):
			poses = [pose_pred_homo]

			image_full = realtime_reproj(bgr, poses, bbox3d, K_full, colors=['g'])
			x1, y1, x2, y2 = bbox
			cv2.rectangle(image_full, (x1, y1), (x2, y2), (255,0,0), 2)
			result = draw_keypoints(image_full, validcorners, K_full, K_crop)
			result = draw_keypoints(result, notvalidcorners, K_full, K_crop, color=(0, 0, 255))
		else:
			result = bgr
		    
		# display processed frame
		result = np.concatenate((result ,dpt), axis = 0)
		cv2.imshow('frame', result)
		#writer.write(result)
		print('')

		# detect 'q' key to exit the loop
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# close realsense pipeline and destroy all windows
	pipe.stop()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
