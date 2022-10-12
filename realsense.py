import cv2
import torch
import hydra
from tqdm import tqdm
import os.path as osp
import numpy as np
import pyrealsense2 as rs
import open3d as o3d

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
	# initialize realsense, open3d and opencv configurations
	pipe = rs.pipeline()
	rscfg = rs.config()
	width, height = 1280, 800
	rscfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
	rscfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
	profile = pipe.start(rscfg)
	depth_sensor = profile.get_device().first_depth_sensor()
	depth_scale = depth_sensor.get_depth_scale()
	clipping_distance_in_meters = 1
	clipping_distance = clipping_distance_in_meters / depth_scale
	
	align = rs.align(rs.stream.color)
	colorizer = rs.colorizer()
	treg = o3d.t.pipelines.registration
	Float32 = o3d.core.Dtype.Float32
	
	pcd = o3d.geometry.PointCloud() # viz rgbd pcd
	mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
	mesh.scale(0.1, center = mesh.get_center())
	
	# cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	vis = o3d.visualization.Visualizer()
	vis.create_window()
	
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
	sfm = o3d.io.read_point_cloud(osp.join(anno_dir,'..','obj_01.ply'))
	#sfm = o3d.io.read_point_cloud(osp.join(anno_dir,'..','model.ply'))
	#sfm.estimate_normals()
	sfmpcd = o3d.geometry.PointCloud() # viz syn pcd

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
	intrin_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, intrin.fx, intrin.fy, intrin.ppx, intrin.ppy)
	K_full = np.array([[intrin.fx, 0, intrin.ppx],
					   [0, intrin.fy, intrin.ppy],
					   [0,         0,          1]])
	
	# abstract object detection pipeline
	def detect_object(inp, init):
		if init == False:
			print('initial object-detection frame')
			bbox, inp_crop, K_crop = yolov5_detector.detect(inp, K_full, crop_size=512)
			init = True
		else:
			if len(previous_inliers) < 10:
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
		
	# abstract iterative matching pipeline
	def alignICP(pcd,sfmpcd):
		init_source_to_target = o3d.core.Tensor(np.eye(4))
		source = o3d.t.geometry.PointCloud.from_legacy(sfmpcd).cuda(0)
		target = o3d.t.geometry.PointCloud.from_legacy(pcd).cuda(0)
		estimation = treg.TransformationEstimationPointToPoint()
		criteria_list = [treg.ICPConvergenceCriteria(0.0001, 0.0001, 20),
						 treg.ICPConvergenceCriteria(0.0001, 0.0001, 15),
						 treg.ICPConvergenceCriteria(0.0001, 0.0001, 10)]
		max_correspondence_distances = o3d.utility.DoubleVector([0.01, 0.01, 0.01])
		voxel_sizes = o3d.utility.DoubleVector([0.002, 0.0015, 0.001])
		reg_multiscale_icp = treg.multi_scale_icp(source, target, voxel_sizes,
												  criteria_list,
												  max_correspondence_distances,
												  init_source_to_target, estimation)
		print('     fitness :', reg_multiscale_icp.fitness)
		print('     rmse :', reg_multiscale_icp.inlier_rmse)
		return reg_multiscale_icp.transformation.numpy()
	
	# abstract iterative matching pipeline
	def alignColorICP(src, tgt):
		init_source_to_target = o3d.core.Tensor(np.eye(4))
		source = o3d.t.geometry.PointCloud.from_legacy(src).cuda(0)
		target = o3d.t.geometry.PointCloud.from_legacy(tgt).cuda(0)
		loss = treg.robust_kernel.RobustKernel(treg.robust_kernel.RobustKernelMethod.TukeyLoss,
											   0.01)
		estimation = treg.TransformationEstimationForColoredICP(loss)
		criteria_list = [treg.ICPConvergenceCriteria(0.001, 0.001, 20),
						 treg.ICPConvergenceCriteria(0.001, 0.001, 20),
						 treg.ICPConvergenceCriteria(0.001, 0.001, 20)]
		max_correspondence_distances = o3d.utility.DoubleVector([0.03, 0.02, 0.01])
		voxel_sizes = o3d.utility.DoubleVector([0.002, 0.0015, 0.001])
		reg_multiscale_icp = treg.multi_scale_icp(source, target, voxel_sizes,
												  criteria_list,
												  max_correspondence_distances,
												  init_source_to_target, estimation)
		return reg_multiscale_icp.transformation.numpy()
		
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
		dpt = np.asanyarray(depth_frame.get_data())
		rgb = np.asanyarray(color_frame.get_data())
		bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
		frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) #for onepose
		rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
				o3d.geometry.Image(bgr),
				o3d.geometry.Image(dpt),
				depth_scale=1.0 / depth_scale,
				depth_trunc=clipping_distance_in_meters,
				convert_rgb_to_intensity=False)
		temp = o3d.geometry.PointCloud.create_from_rgbd_image(
				rgbd_image, intrin_o3d)
		pcd.points = temp.points
		pcd.colors = temp.colors
		if init == False:
			vis.add_geometry(pcd)
			vis.add_geometry(sfmpcd)
			vis.add_geometry(mesh)
		
		##### object detection
		object_detected, bbox, inp_crop, K_crop, init = detect_object(frame, init)
		
		if object_detected:
			##### process the frame
			print(inp_crop.shape)
			pose_pred_homo, inliers, _, _ = onePoseForwardPass(inp_crop, K_crop)
			previous_frame_pose = pose_pred_homo
			previous_inliers = inliers
		else:
			##### Reset stored poses
			previous_frame_pose = np.eye(4)
			previous_inliers = []

		if object_detected and not np.array_equal(pose_pred_homo, np.eye(4)):
			sfm.transform(pose_pred_homo)
			bbox3d_t = pose_pred_homo @ np.concatenate((bbox3d,np.ones(8).reshape(8,1)), axis=1).T
			bbox3d_t = (bbox3d_t[:3] / bbox3d_t[3]).T
			bbo3d = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox3d_t))
			temp_crop = pcd.crop(bbo3d)
			pcd.points = temp_crop.points
			pcd.colors = temp_crop.colors
			refine_homo = alignColorICP(pcd, sfm) # icp
			#refine_homo = alignICP(pcd, sfm)
			sfm.transform(np.linalg.inv(refine_homo)) # icp
			sfmpcd.points = sfm.points
			sfmpcd.colors = sfm.colors
			#sfmpcd.paint_uniform_color([1,0.706,0])
			vis.update_geometry(sfmpcd) # update location of render
			sfm.transform(refine_homo) # icp
			sfm.transform(np.linalg.inv(pose_pred_homo))
		vis.update_geometry(pcd) # update rgbd cloud
		vis.poll_events()
		vis.update_renderer()

	# close realsense pipeline and destroy all windows
	pipe.stop()
	vis.destroy_window()

if __name__ == "__main__":
    main()
