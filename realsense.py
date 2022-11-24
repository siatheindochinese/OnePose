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
	
def transform_pts(pts, homo):
	# pts: 3D pts, nx3 numpy array
	# homo: P3 transform, 4x4 numpy array
	pts_t = homo @ np.concatenate((pts, np.ones(8).reshape(8,1)), axis=1).T
	return (pts_t[:3] / pts_t[3]).T
	
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
def projective_angle(pose_pred_homo, bbox3d, K_full):
	centroid = K_full @ pose_pred_homo[:3,3].reshape(3,1)
	centroid = (centroid/centroid[2,0])[:2].astype(int).squeeze()
	top_bb = vis_utils.reproj(K_full, pose_pred_homo, np.average(bbox3d[[1,2,5,6]],axis=0)).astype(int).squeeze()
	top_aligned = np.array([centroid[0],0])
	v0 = top_aligned - centroid
	v1 = top_bb - centroid
	rot2d = angle_between(v0, v1) * 180 / np.pi
	if v1[0] < v0[0]:
		rot2d = 360 - rot2d
	return rot2d, centroid, top_bb, top_aligned
	
@torch.no_grad()
@hydra.main(config_path='configs/', config_name='config.yaml')
def main(cfg):
	##################################################
	# initialize realsense and open3d configurations #
	##################################################
	# init realsense camera
	pipe = rs.pipeline()
	rscfg = rs.config()
	width, height = 1280, 800
	rscfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
	rscfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
	profile = pipe.start(rscfg)
	
	# gretting realsense depth attributes
	depth_sensor = profile.get_device().first_depth_sensor()
	depth_sensor.set_option(rs.option.laser_power,360)
	depth_scale = depth_sensor.get_depth_scale()
	clipping_distance_in_meters = 1
	clipping_distance = clipping_distance_in_meters / depth_scale
	
	# realsense utility functions
	align = rs.align(rs.stream.color)
	spatial = rs.spatial_filter()
	hole_filling = rs.hole_filling_filter(0)
	colorizer = rs.colorizer()
	treg = o3d.t.pipelines.registration
	Float32 = o3d.core.Dtype.Float32
	
	# init open3d objects for visualization
	pcd = o3d.geometry.PointCloud() # viz rgbd pcd
	xyz = o3d.geometry.TriangleMesh.create_coordinate_frame()
	xyz.scale(0.1, center = xyz.get_center())
	
	# init open3d visualizer
	vis = o3d.visualization.Visualizer()
	vis.create_window()
	
	# init opencv window
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame', width, height)
	cv2.namedWindow('cropped',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('cropped', 512, 512)

	##############################################
	# load SuperPoint, SuperGlue and OnePose GAT #
	##############################################
	matching_model, extractor_model = load_model(cfg)
	
	########################
	# load yolov5 detector #
	########################
	yolov5_detector = YoloV5Detector(cfg.yolov5_dir, cfg.yolov5_weights_dir)

	############
	# load SfM #
	############
	anno_dir = osp.join(cfg.sfm_model_dir, f'outputs_{cfg.network.detection}_{cfg.network.matching}', 'anno')
	avg_anno_3d_path = osp.join(anno_dir, 'anno_3d_average.npz')
	clt_anno_3d_path = osp.join(anno_dir, 'anno_3d_collect.npz')
	idxs_path = osp.join(anno_dir, 'idxs.npy')
	sfm = o3d.io.read_point_cloud(osp.join(anno_dir,'..','3Dmodel.ply')) # open3d object to compute transformations
	sfm.scale(1000, center=sfm.get_center())
	sfmpcd = o3d.geometry.PointCloud() # open3d object for visualization

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

	###################
	# load intrinsics #
	###################
	# getting realsense intrinsics
	intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
	intrin_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, intrin.fx, intrin.fy, intrin.ppx, intrin.ppy)
	K_full = np.array([[intrin.fx, 0, intrin.ppx],
					   [0, intrin.fy, intrin.ppy],
					   [0,         0,          1]])
					   
	
	# init open3d OffscreenRenderer
	renderer_pc = o3d.visualization.rendering.OffscreenRenderer(width, height)
	renderer_pc.setup_camera(o3d.camera.PinholeCameraIntrinsic(width, height, intrin.fx, intrin.fy, intrin.ppx, intrin.ppy),
							 np.eye(4))
	renderer_pc.scene.set_background(np.array([0, 0, 0, 0]))
	mat = o3d.visualization.rendering.MaterialRecord()
	mat.shader = 'defaultUnlit'
	
	#########################
	# Some while-loop flags #
	#########################
	init_obj_detect = False
	previous_frame_pose = np.eye(4)
	previous_angle = 0
	previous_inliers = 0
	
	######################################
	# abstract object detection pipeline #
	######################################
	def detect_object(inp, init_obj_detect):
		if init_obj_detect == False:
			print('initial object-detection frame')
			bbox, inp_crop, K_crop = yolov5_detector.detect(inp, K_full, crop_size=512)
			init_obj_detect = True
		else:
			if previous_inliers <= 10:
				print('object-detection frame')
				bbox, inp_crop, K_crop = yolov5_detector.detect(inp, K_full, crop_size=512)
			else:
				print('GT frame')
				bbox, inp_crop, K_crop = yolov5_detector.previous_pose_detect(inp, K_full, previous_frame_pose, bbox3d, crop_size=512)
		print('bbox=',bbox)
		##### Determine if object is detected within the frame
		object_detected = not (bbox == np.array([0, 0, height, width])).all() #hardcoded dimensions
		print('obj_detected =',object_detected)
		return object_detected, bbox, inp_crop, K_crop, init_obj_detect
		
	#############################
	# abstract OnePose pipeline #
	#############################
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
		
	############################################
	# abstract angle-adjusted OnePose pipeline #
	############################################
	def onePoseAngledForwardPass(inp_crop, K_crop, angle):
		rotmat = cv2.getRotationMatrix2D((256,256), angle, 1)
		rotmat_homo = np.concatenate((rotmat, np.array([[0,0,1]])),axis=0)
		rotmatinv = np.linalg.inv(rotmat_homo)[:2]
		rot_crop = cv2.warpAffine(inp_crop, rotmat, (inp_crop.shape[0], inp_crop.shape[1]))
		inp_crop_cuda = torch.from_numpy(rot_crop.astype(np.float32)[None][None]/255.).cuda()
		pred_detection = extractor_model(inp_crop_cuda)
		pred_detection = {k: v[0].detach().cpu().numpy() for k, v in pred_detection.items()}
		kpts_new = np.concatenate((pred_detection['keypoints'], np.ones(len(pred_detection['keypoints'])).astype(int).reshape(-1,1)),axis=1)
		kpts_new = rotmatinv @ kpts_new.T
		pred_detection['keypoints'] = kpts_new.T.astype(int)
		inp_data = pack_data(avg_descriptors3d, clt_descriptors, keypoints3d, pred_detection, np.array([height, width]))
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
		inliers = len(inliers)

		return pose_pred_homo, inliers, validcorners, notvalidcorners
		
	############################################
	# abstract multiscale vanilla ICP pipeline #
	############################################
	def alignmultiScaleICP(src,tgt):
		init_source_to_target = o3d.core.Tensor(np.eye(4))
		source = o3d.t.geometry.PointCloud.from_legacy(src).cuda(0)
		target = o3d.t.geometry.PointCloud.from_legacy(tgt).cuda(0)
		loss = treg.robust_kernel.RobustKernel(treg.robust_kernel.RobustKernelMethod.TukeyLoss,
											   0.005)
		estimation = treg.TransformationEstimationPointToPlane(loss)
		criteria_list = [treg.ICPConvergenceCriteria(0.0001, 0.0001, 20),
						 treg.ICPConvergenceCriteria(0.0001, 0.0001, 15),
						 treg.ICPConvergenceCriteria(0.0001, 0.0001, 10)]
		max_correspondence_distances = o3d.utility.DoubleVector([0.01, 0.01, 0.01])
		voxel_sizes = o3d.utility.DoubleVector([0.002, 0.0015, 0.001])
		reg_multiscale_icp = treg.multi_scale_icp(source, target, voxel_sizes,
												  criteria_list,
												  max_correspondence_distances,
												  init_source_to_target, estimation)
												  
		del source
		del target
		return reg_multiscale_icp.transformation.numpy(), reg_multiscale_icp.fitness
		
	#################################
	# abstract vanilla ICP pipeline #
	#################################
	def alignICP(src, tgt):
		init_source_to_target = o3d.core.Tensor(np.eye(4))
		source = o3d.t.geometry.PointCloud.from_legacy(src).cuda(0)
		target = o3d.t.geometry.PointCloud.from_legacy(tgt).cuda(0)
		loss = treg.robust_kernel.RobustKernel(treg.robust_kernel.RobustKernelMethod.TukeyLoss,
											   0.01)
		estimation = treg.TransformationEstimationPointToPoint()
		criteria = treg.ICPConvergenceCriteria(0.001, 0.001, 20)
		max_correspondence_distance = 0.01
		voxel_size = 0.001
		reg_icp = treg.icp(source, target,
									  max_correspondence_distance,
									  init_source_to_target, estimation,
									  criteria, voxel_size)
		del source
		del target
		return reg_icp.transformation.numpy(), reg_icp.fitness
	
	############################################
	# abstract multiscale colored ICP pipeline #
	############################################
	def alignMultiScaleColorICP(src, tgt):
		init_source_to_target = o3d.core.Tensor(np.eye(4))
		source = o3d.t.geometry.PointCloud.from_legacy(src).cuda(0)
		target = o3d.t.geometry.PointCloud.from_legacy(tgt).cuda(0)
		loss = treg.robust_kernel.RobustKernel(treg.robust_kernel.RobustKernelMethod.TukeyLoss,
											   0.01)
		estimation = treg.TransformationEstimationForColoredICP(loss)
		criteria_list = [treg.ICPConvergenceCriteria(0.001, 0.001, 20),
						 treg.ICPConvergenceCriteria(0.0001, 0.0001, 20),
						 treg.ICPConvergenceCriteria(0.00001, 0.00001, 10)]
		max_correspondence_distances = o3d.utility.DoubleVector([0.03, 0.02, 0.01])
		voxel_sizes = o3d.utility.DoubleVector([0.003, 0.002, 0.001])
		reg_multiscale_icp = treg.multi_scale_icp(source, target, voxel_sizes,
												  criteria_list,
												  max_correspondence_distances,
												  init_source_to_target, estimation)
		del source
		del target
		return reg_multiscale_icp.transformation.numpy(), reg_multiscale_icp.fitness
		
	#################################
	# abstract colored ICP pipeline #
	#################################
	def alignColorICP(src, tgt):
		init_source_to_target = o3d.core.Tensor(np.eye(4))
		source = o3d.t.geometry.PointCloud.from_legacy(src).cuda(0)
		target = o3d.t.geometry.PointCloud.from_legacy(tgt).cuda(0)
		loss = treg.robust_kernel.RobustKernel(treg.robust_kernel.RobustKernelMethod.TukeyLoss,
											   0.01)
		estimation = treg.TransformationEstimationForColoredICP(loss)
		criteria = treg.ICPConvergenceCriteria(0.0001, 0.0001, 50)
		max_correspondence_distance = 0.01
		voxel_size = 0.001
		reg_icp = treg.icp(source, target,
									  max_correspondence_distance,
									  init_source_to_target, estimation,
									  criteria, voxel_size)
		del source
		del target
		return reg_icp.transformation.numpy(), reg_icp.fitness
		
	###############################
	# depth point cloud from mesh #
	###############################
	def zbuf_from_geometry(sfm):
		print('points :', len(sfm.points))
		renderer_pc.scene.add_geometry('sfmpcd', sfm, mat)
		dpt = renderer_pc.render_to_depth_image(z_in_view_space = True)
		cv2.imshow('syn', np.array(dpt))
		renderer_pc.scene.clear_geometry()
		dptpcd = o3d.geometry.PointCloud.create_from_depth_image(dpt, intrin_o3d)
		return dptpcd
		
	########################
	# general ICP function #
	########################
	def refineICP(pose_pred_homo):
		sfm.transform(pose_pred_homo)
		##### render raw pcd
		dptpcd = zbuf_from_geometry(sfm)
		##### transform 3d bbox to predicted pose
		bbox3d_t = transform_pts(bbox3d, pose_pred_homo)
		bbo3d = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox3d_t))
		##### crop rgbd pcd with 3d bbox
		temp_crop = pcd.crop(bbo3d)
		#pcd.points = temp_crop.points #crop for viz
		#pcd.colors = temp_crop.colors #crop for viz
		##### ICP refine algorithm
		refine_homo, fitness = alignICP(temp_crop, dptpcd) # icp
		##### refine pose with predicted icp transform
		sfm.transform(np.linalg.inv(refine_homo)) # icp
		dptpcd.transform(np.linalg.inv(refine_homo))
		##### update syn pcd
		sfmpcd.points = dptpcd.points
		sfmpcd.colors = dptpcd.colors
		vis.update_geometry(sfmpcd) # update location of render
		##### untransform icp refine and pose
		sfm.transform(refine_homo)
		sfm.transform(np.linalg.inv(pose_pred_homo))
		return np.linalg.inv(refine_homo) @ pose_pred_homo
	
	#############
	# Main Loop #
	#############
	while True:
		print('##############')
		print('# NEXT FRAME #')
		print('##############')
		print('init_obj_detect:', init_obj_detect, 'previous_inliers:', previous_inliers)
		# stream the next frame
		# convert to RGBD image
		# convert to colored point-cloud
		frameset = pipe.wait_for_frames()
		frameset = align.process(frameset)
		color_frame = frameset.get_color_frame()
		depth_frame = frameset.get_depth_frame()
		filtered_depth = spatial.process(depth_frame)
		dpt = np.asanyarray(filtered_depth.get_data())
		rgb = np.asanyarray(color_frame.get_data())
		bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
		frame = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) #for onepose
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
		if init_obj_detect == False:
			vis.add_geometry(pcd)
			vis.add_geometry(sfmpcd)
			vis.add_geometry(xyz)
			
			ctr = vis.get_view_control()
			ctr.set_front((0, 0, -1))
			ctr.set_up((0, -1, 0))
		
		# object detection
		object_detected, bbox, inp_crop, K_crop,\
		 init_obj_detect = detect_object(frame, init_obj_detect)
		
		# pose estimation (if object detected)
		if object_detected:
			##### process the frame
			#pose_pred_homo, inliers, _, _ = onePoseForwardPass(inp_crop, K_crop) without angle fix
			pose_pred_homo, inliers, validcorners, notvalidcorners = onePoseAngledForwardPass(inp_crop, K_crop, previous_angle)
			rot2d, centroid, top_bb, top_aligned = projective_angle(pose_pred_homo, bbox3d, K_full)
			
			##### use ICP if possible
			if False:
				pose_pred_homo = refineICP(pose_pred_homo)
			else:
				sfm.transform(pose_pred_homo)
				#dptpcd = zbuf_from_geometry(sfm)
				#sfmpcd.points = dptpcd.points
				#sfmpcd.colors = dptpcd.colors
				sfmpcd.points = sfm.points
				sfmpcd.colors = sfm.colors
				vis.update_geometry(sfmpcd)
				sfm.transform(np.linalg.inv(pose_pred_homo))
			
			##### processing for viz
			poses = [pose_pred_homo]
			rot_adjust = cv2.getRotationMatrix2D(tuple(centroid), rot2d, 1)
			image_full = realtime_reproj(rgb, poses, bbox3d, K_full, colors=['g'])
			x1, y1, x2, y2 = bbox
			cv2.line(image_full, tuple(centroid), tuple(top_bb),(255,255,255),2)
			cv2.line(image_full, tuple(centroid), (centroid[0], y1),(255,255,255),2)
			cv2.rectangle(image_full, (x1, y1), (x2, y2), (255,0,0), 2)
			result = draw_keypoints(image_full, validcorners, K_full, K_crop)
			result = draw_keypoints(result, notvalidcorners, K_full, K_crop, color=(0, 0, 255))
			rot_mat = cv2.getRotationMatrix2D((256,256), rot2d, 1)
			result_crop = cv2.warpAffine(inp_crop, rot_mat, (inp_crop.shape[0], inp_crop.shape[1]))
			
			#### temporal variable update
			previous_frame_pose = pose_pred_homo
			previous_inliers = inliers
			previous_angle = rot2d
		else:
			print('     No object Detected, Reset pose, fitness and flag')
			result = rgb
			result_crop = frame
			##### Reset temporal variables
			previous_frame_pose = np.eye(4)
			previous_inliers = 0
			previous_angle = 0

		# open3d viz
		vis.update_geometry(pcd) # update rgbd cloud
		vis.poll_events()
		vis.update_renderer()
		
		# opencv viz
		cv2.imshow('frame', result)
		#writer.write(result)
		cv2.imshow('cropped', result_crop)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		print('')
		print('')

	# close realsense pipeline and destroy all windows
	pipe.stop()
	vis.destroy_window()
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
