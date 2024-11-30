#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import mmcv
import time
import rclpy
import copy

import torch
import mmengine
import threading
import numpy as np
import onnxruntime
from typing import List
from abc import ABCMeta, abstractmethod
from rclpy.executors import MultiThreadedExecutor

from ament_index_python.packages import get_package_share_directory

from mmengine import DictAction, Config
from mmdet.structures.bbox import bbox2roi
from mmaction.apis import detection_inference
from mmaction.utils import frame_extract, get_str_type
from mmdet.apis import inference_detector, init_detector

from mmengine.structures import InstanceData
from mmaction.structures import ActionDataSample

from cv_bridge import CvBridge
from cv_bridge_util.cv_bridge import CvBridgeUtils
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from rclpy.node import Node
# from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown
# from mmpose.apis import vis_pose_result
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances

from visualization_msgs.msg import Marker, MarkerArray
from mmlab_ros_msgs.msg import Ax3DPose, AxKeyPoint, Ax3DPoseWithLabel, Ax3DPoseWithLabelArray
from rclpy.time import Time
from rclpy.duration import Duration

from typing import Dict, List, Optional, Tuple

from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from image_geometry import PinholeCameraModel

MM_ACTION_NOSE = 0
MM_ACTION_LEFT_EYE = 1
MM_ACTION_RIGHT_EYE = 2
MM_ACTION_LEFT_EAR = 3
MM_ACTION_RIGHT_EAR = 4
MM_ACTION_LEFT_SHOULDER = 5
MM_ACTION_RIGHT_SHOULDER = 6
MM_ACTION_LEFT_ELBOW = 7
MM_ACTION_RIGHT_ELBOW = 8
MM_ACTION_LEFT_WRIST = 9
MM_ACTION_RIGHT_WRIST = 10
MM_ACTION_LEFT_HIP = 11
MM_ACTION_RIGHT_HIP = 12
MM_ACTION_LEFT_KNEE = 13
MM_ACTION_RIGHT_KNEE = 14
MM_ACTION_LEFT_ANKLE = 15
MM_ACTION_RIGHT_ANKLE = 16


class MmActionDetector(Node):
    def __init__(self) -> None:
        super().__init__("mmaction_node")
        self.package_dir = get_package_share_directory("mmlab_ros_node")
        self.get_logger().info(f"your shared package dir is: {self.package_dir}")

        ############################################
        # ロボットに依存するパラメータ
        ############################################
        self.img_width = 640
        self.img_height = 480
        self.img_topic_hz = 30

        self._p_offset_x = 0.0
        self._p_offset_y = 0.0
        self._p_offset_z = 0.0

        ############################################
        # MMAction2固有のパラメータ
        ############################################
        self.short_side = 256
        self.predict_stepsize = 8
        self.step_size = 16  # xフレームに一回の推論
        self.max_labels_per_bbox = 5
        self.pose_th = 0.4

        self.mm_keypoint_info = {
            "nose": {"id": 0, "color": [51, 153, 255], "type": 'upper', "swap": None},
            "left_eye": {"id": 1, "color": [51, 153, 255], "type": 'upper', "swap": 'right_eye'},
            "right_eye": {"id": 2, "color": [51, 153, 255], "type": 'upper', "swap": 'left_eye'},
            "left_ear": {"id": 3, "color": [51, 153, 255], "type": 'upper', "swap": 'right_ear'},
            "right_ear": {"id": 4, "color": [51, 153, 255], "type": 'upper', "swap": 'left_ear'},
            "left_shoulder": {"id": 5, "color": [0, 255, 0], "type": 'upper', "swap": 'right_shoulder'},
            "right_shoulder": {"id": 6, "color": [255, 128, 0], "type": 'upper', "swap": 'left_shoulder'},
            "left_elbow": {"id": 7, "color": [0, 255, 0], "type": 'upper', "swap": 'right_elbow'},
            "right_elbow": {"id": 8, "color": [255, 128, 0], "type": 'upper', "swap": 'left_elbow'},
            "left_wrist": {"id": 9, "color": [0, 255, 0], "type": 'upper', "swap": 'right_wrist'},
            "right_wrist": {"id": 10, "color": [255, 128, 0], "type": 'upper', "swap": 'left_wrist'},
            "left_hip": {"id": 11, "color": [0, 255, 0], "type": 'lower', "swap": 'right_hip'},
            "right_hip": {"id": 12, "color": [255, 128, 0], "type": 'lower', "swap": 'left_hip'},
            "left_knee": {"id": 13, "color": [0, 255, 0], "type": 'lower', "swap": 'right_knee'},
            "right_knee": {"id": 14, "color": [255, 128, 0], "type": 'lower', "swap": 'left_knee'},
            "left_ankle": {"id": 15, "color": [0, 255, 0], "type": 'lower', "swap": 'right_ankle'},
            "right_ankle": {"id": 16, "color": [255, 128, 0], "type": 'lower', "swap": 'left_ankle'}
        }

        self.mm_skeleton_info = {
            "left_ankle2left_knee": {"link": ('left_ankle', 'left_knee'), "id": 0, "color": [0, 255, 0]},
            "left_knee2left_hip": {"link": ('left_knee', 'left_hip'), "id": 1, "color": [0, 255, 0]},
            "right_ankle2right_knee": {"link": ('right_ankle', 'right_knee'), "id": 2, "color": [255, 128, 0]},
            "right_knee2right_hip": {"link": ('right_knee', 'right_hip'), "id": 3, "color": [255, 128, 0]},
            "left_hip2right_hip": {"link": ('left_hip', 'right_hip'), "id": 4, "color": [51, 153, 255]},
            "left_shoulder2left_hip": {"link": ('left_shoulder', 'left_hip'), "id": 5, "color": [51, 153, 255]},
            "right_shoulder2right_hip": {"link": ('right_shoulder', 'right_hip'), "id": 6, "color": [51, 153, 255]},
            "left_shoulder2right_shoulder": {"link": ('left_shoulder', 'right_shoulder'), "id": 7, "color": [51, 153, 255]},
            "left_shoulder2left_elbow": {"link": ('left_shoulder', 'left_elbow'), "id": 8, "color": [0, 255, 0]},
            "right_shoulder2right_elbow": {"link": ('right_shoulder', 'right_elbow'), "id": 9, "color": [255, 128, 0]},
            "left_elbow2left_wrist": {"link": ('left_elbow', 'left_wrist'), "id": 10, "color": [0, 255, 0]},
            "right_elbow2right_wrist": {"link": ('right_elbow', 'right_wrist'), "id": 11, "color": [255, 128, 0]},
            "left_eye2right_eye": {"link": ('left_eye', 'right_eye'), "id": 12, "color": [51, 153, 255]},
            "nose2left_eye": {"link": ('nose', 'left_eye'), "id": 13, "color": [51, 153, 255]},
            "nose2right_eye": {"link": ('nose', 'right_eye'), "id": 14, "color": [51, 153, 255]},
            "left_eye2left_ear": {"link": ('left_eye', 'left_ear'), "id": 15, "color": [51, 153, 255]},
            "right_eye2right_ear": {"link": ('right_eye', 'right_ear'), "id": 16, "color": [51, 153, 255]},
            "left_ear2left_shoulder": {"link": ('left_ear', 'left_shoulder'), "id": 17, "color": [51, 153, 255]},
            "right_ear2right_shoulder": {"link": ('right_ear', 'right_shoulder'), "id": 18, "color": [51, 153, 255]}
        }

        # resize frames to shortside
        self.new_w, self.new_h = mmcv.rescale_size((self.img_width, self.img_height), (self.short_side, np.Inf))
        self.w_ratio, self.h_ratio = self.new_w / self.img_width, self.new_h / self.img_height

        ############################################
        # MMAction2のモデルパスなど (PCに依存)
        ############################################
        self.action_score_thr = 0.15
        self.det_score_thr = 0.7  # detection score threshold
        self.det_cat_id = 0  # detection category is (0 = human)
        self.device = "cuda:0"

        # label map
        self.label_map_path = "config/mmaction/label_map/ava_label_map.txt"
        self.label_map_path = os.path.join(self.package_dir, self.label_map_path)
        self.get_logger().info(f"your label map path is {self.label_map_path}")
        self.label_map = self.load_label_map(self.label_map_path)

        # detection モデルパス
        self.detection_pth_path = "config/ckpt/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
        self.detection_config = "config/mmdetection/faster_rcnn/faster-rcnn_r50_fpn_2x_coco_infer.py"

        self.detection_pth_path = os.path.join(self.package_dir, self.detection_pth_path)
        self.detection_config = os.path.join(self.package_dir, self.detection_config)
        self.detection_model = init_detector(self.detection_config, self.detection_pth_path, self.device)

        # 骨格検出モデル
        self.pose_pth_path = "config/ckpt/seresnet50_coco_256x192-25058b66_20200727.pth"
        self.pose_cfg_path = "config/mmpose/seresnet/td-hm_seresnet50_8xb64-210e_coco-256x192.py"
        self.pose_pth_path = os.path.join(self.package_dir, self.pose_pth_path)
        self.pose_cfg_path = os.path.join(self.package_dir, self.pose_cfg_path)
        self.pose_estimator = init_pose_estimator(self.pose_cfg_path, self.pose_pth_path, device=self.device)

        # build visualizer
        skeleton_style = "mmpose"  # or "openpose"
        self.pose_estimator.cfg.visualizer.radius = 2
        self.pose_estimator.cfg.visualizer.alpha = 0.2
        self.pose_estimator.cfg.visualizer.line_width = 2
        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        # the dataset_meta is loaded from the checkpoint and
        # then pass to the model in init_pose_estimator
        self.visualizer.set_dataset_meta(self.pose_estimator.dataset_meta, skeleton_style=skeleton_style)

        # mmaction2モデルパス
        self.action_onnx_path = "config/ckpt/videomae.onnx"
        self.action_config = "config/mmaction/detection/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py"

        self.action_onnx_path = os.path.join(self.package_dir, self.action_onnx_path)
        self.action_config = os.path.join(self.package_dir, self.action_config)
        self.session = onnxruntime.InferenceSession(self.action_onnx_path)

        # self.display_width = 700
        # self.display_height = 500
        # self.show = False

        # Get clip_len, frame_interval and calculate center index of each clip
        self.config = mmengine.Config.fromfile(self.action_config)
        self.img_norm_cfg = dict(
            mean=np.array(self.config.model.data_preprocessor.mean),
            std=np.array(self.config.model.data_preprocessor.std),
            to_rgb=False
        )

        # config.merge_from_dict(args.cfg_options)
        # val_pipeline = config.val_pipeline

        # sampler = [x for x in val_pipeline if get_str_type(x['type']) == 'SampleAVAFrames'][0]
        # clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
        # window_size = clip_len * frame_interval
        # assert clip_len % 2 == 0, 'We would like to have an even clip_len'
        # # Note that it's 1 based here
        # timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
        #                     args.predict_stepsize)

        ############################################
        # ros interface
        ############################################
        self.bridge = CvBridge()
        self._bridge = CvBridgeUtils(self)
        # self.img_topic_name = "/camera/rgb/image_raw"

        custom_qos_profile = QoSProfile(depth=1, reliability=qos_profile_sensor_data.reliability, durability=qos_profile_sensor_data.durability, history=qos_profile_sensor_data.history)

        self._rgb_msg = None
        self._depth_msg = None
        self.cv_img = None
        self.cv_depth = None

        # camera modelの初期化(for orbbec)
        # self.camera_tf_frame = "camera_link"
        # self.camera_model = PinholeCameraModel()
        # self._camera_info_sub = self.create_subscription(CameraInfo, "/camera/color/camera_info", self.camera_info_callback, custom_qos_profile)
        # self._rgb_sub = self.create_subscription(Image, "/camera/color/image_raw", self.rgb_callback, custom_qos_profile)
        # self._depth_sub = self.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, custom_qos_profile)

        # camera modelの初期化(for HSRB)
        self.camera_tf_frame = "head_rgbd_sensor_rgb_frame"
        self.camera_model = PinholeCameraModel()
        self._camera_info_sub = self.create_subscription(CameraInfo, "/head_rgbd_sensor/rgb/camera_info", self.camera_info_callback, custom_qos_profile)
        self._rgb_sub = self.create_subscription(CompressedImage, "/head_rgbd_sensor/rgb/image_rect_color/compressed", self.rgb_callback, custom_qos_profile)
        self._depth_sub = self.create_subscription(CompressedImage, "head_rgbd_sensor/depth_registered/image_rect_raw/compressedDepth", self.depth_callback, custom_qos_profile)

        self._pub_result_action_img = self.create_publisher(Image, "~/image/action", qos_profile_sensor_data)
        self._pub_result_pose_img = self.create_publisher(Image, "~/image/pose", qos_profile_sensor_data)
        self._pub_result_pose_marker = self.create_publisher(MarkerArray, "~/result_skeleton/marker_array", qos_profile_sensor_data)
        self._pub_ax_3d_poses = self.create_publisher(Ax3DPoseWithLabelArray, "~/people_poses", qos_profile_sensor_data)

        self.timer = self.create_timer(0.03, self.pose_estimation_callback)
        self.use_action_recognition = False

        if self.use_action_recognition:
            self.timer = self.create_timer(0.05, self.action_recognition_callback)

        # プログラムに必要な変数
        self.original_frames = []
        self.depth_frames = []
        self.human_detections = []
        self.human_poses = []
        self.frame_counter = 0

        self.text_fontface = cv2.FONT_HERSHEY_DUPLEX
        self.text_fontscale = 0.5
        self.text_fontcolor = (255, 255, 255)
        self.text_thickness = 1
        self.text_linetype = 1
        plate = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
        plate = plate.split('-')
        self.plate = [self.hex2color(h) for h in plate]

    def rgb_callback(self, msg: Image):
        self.get_logger().debug("get rgb image")
        self.cv_img = self._bridge.compressed_imgmsg_to_cv2(msg)

    def depth_callback(self, msg: Image):
        self.get_logger().debug("get depth image")
        self.cv_depth = self._bridge.compressed_imgmsg_to_depth(msg)

    def camera_info_callback(self, msg: CameraInfo):
        # CameraInfoメッセージからカメラモデルを設定
        self.camera_model.fromCameraInfo(msg)
        self.get_logger().info("Camera model parameters set.")

        # 使用例: カメラの焦点距離や光学中心の取得
        fx = self.camera_model.fx()
        fy = self.camera_model.fy()
        cx = self.camera_model.cx()
        cy = self.camera_model.cy()
        self.get_logger().info(f"Camera focal lengths: fx={fx}, fy={fy}")
        self.get_logger().info(f"Camera optical center: cx={cx}, cy={cy}")
        self.destroy_subscription(self._camera_info_sub)

    def _pose_visualizer(self, image, data_samples):
        self.visualizer.add_datasample(
            'result',
            image,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=True,
            draw_bbox=True,
            show_kpt_idx=False,
            skeleton_style="mmpose",
            show=False,
            wait_time=0,
            kpt_thr=0.4)

    def hex2color(self, h):
        """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
        return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))

    def delete(self):
        """デストラクタ
        """
        torch.cuda.empty_cache()
        return True

    def load_label_map(self, file_path: str):
        """Load Label Map.

        Args:
            file_path (str): The file path of label map.
        Returns:
            dict: The label map (int -> label name).
        """
        lines = open(file_path).readlines()
        lines = [x.strip().split(': ') for x in lines]
        return {int(x[0]): x[1] for x in lines}

    def human_detection_inference(self, cv_img):
        """Detect human boxes given frame paths.

        Args:
            args (argparse.Namespace): The arguments.
            frame_paths (list[str]): The paths of frames to do detection inference.

        Returns:
            list[np.ndarray]: The human detection results.
        """
        # rosから取得した画像から
        all_result = inference_detector(self.detection_model, cv_img)

        results = []
        for bbox, label, score in zip(all_result.pred_instances.bboxes, all_result.pred_instances.labels, all_result.pred_instances.scores):
            # human以外の検出結果を削除
            if label != self.det_cat_id:
                continue
            # scoreが一定以下のものを削除
            if score <= self.det_score_thr:
                continue

            bbox_np = bbox.to("cpu").numpy().copy()
            results.append(bbox_np)

        return results

    def abbrev(self, name):
        """Get the abbreviation of label name:

        'take (an object) from (a person)' -> 'take ... from ...'
        """
        while name.find('(') != -1:
            st, ed = name.find('('), name.find(')')
            name = name[:st] + '...' + name[ed + 1:]
        return name

    def draw_one_image(self, frame, bboxes, preds):
        """Draw predictions on one image."""
        for bbox, pred in zip(bboxes, preds):
            # draw bbox
            box = bbox.astype(np.int64)
            st, ed = tuple(box[:2]), tuple(box[2:])
            cv2.rectangle(frame, st, ed, (0, 0, 255), 2)

            # draw texts
            for k, (label, score) in enumerate(pred):
                if k >= self.max_labels_per_bbox:
                    break
                text = f'{self.abbrev(label)}: {score:.4f}'
                location = (0 + st[0], 18 + k * 18 + st[1])
                textsize = cv2.getTextSize(text, self.text_fontface,
                                           self.text_fontscale,
                                           self.text_thickness)[0]
                textwidth = textsize[0]
                diag0 = (location[0] + textwidth, location[1] - 14)
                diag1 = (location[0], location[1] + 2)
                cv2.rectangle(frame, diag0, diag1, self.plate[k + 1], -1)
                cv2.putText(frame, text, location, self.text_fontface,
                            self.text_fontscale, self.text_fontcolor,
                            self.text_thickness, self.text_linetype)

        return frame

    def draw_all_images(self, frames, bboxes, preds):
        """Draw predictions on one image."""
        results = []
        for frame in frames:
            for bbox, pred in zip(bboxes, preds):
                # draw bbox
                box = bbox.astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, (0, 0, 255), 2)

                # draw texts
                for k, (label, score) in enumerate(pred):
                    if k >= self.max_labels_per_bbox:
                        break
                    text = f'{self.abbrev(label)}: {score:.4f}'
                    location = (0 + st[0], 18 + k * 18 + st[1])
                    textsize = cv2.getTextSize(text, self.text_fontface,
                                            self.text_fontscale,
                                            self.text_thickness)[0]
                    textwidth = textsize[0]
                    diag0 = (location[0] + textwidth, location[1] - 14)
                    diag1 = (location[0], location[1] + 2)
                    cv2.rectangle(frame, diag0, diag1, self.plate[k + 1], -1)
                    cv2.putText(frame, text, location, self.text_fontface,
                                self.text_fontscale, self.text_fontcolor,
                                self.text_thickness, self.text_linetype)
            results.append(frame)

        return results

    def inference(self, frames: List[np.ndarray], human_detections):
        """
        actionの推論を行う関数
        frames(List[np.ndarray]): 画像が複数枚入っている配列
        """

        for i in range(len(human_detections)):
            det = human_detections[i][0].tolist()
            det = np.array([det]).astype(np.float32)
            det[:, 0:4:2] *= self.w_ratio
            det[:, 1:4:2] *= self.h_ratio
            human_detections[i] = torch.from_numpy(det[:, :4]).to(self.device)

        proposal = human_detections[0]
        imgs = [frame.astype(np.float32) for frame in frames]
        _ = [mmcv.imnormalize_(img, **self.img_norm_cfg) for img in imgs]
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        rois = bbox2roi([proposal])

        input_feed = {
            'input_tensor': input_array,
            'rois': rois.cpu().data.numpy()
        }
        outputs = self.session.run(['cls_score'], input_feed=input_feed)
        logits = outputs[0]
        scores = 1 / (1 + np.exp(-logits))

        prediction = []
        # N proposals
        for i in range(proposal.shape[0]):
            prediction.append([])
        # Perform action score thr
        for i in range(scores.shape[1]):
            if i not in self.label_map:
                continue
            for j in range(proposal.shape[0]):
                if scores[j, i] > self.action_score_thr:
                    prediction[j].append((self.label_map[i], scores[j, i].item()))

        result_imgs = self.draw_all_images(frames=frames, bboxes=det, preds=prediction)
        result_img = result_imgs[-1]
        # for result_img in result_imgs:
        pub_img_msg = self.bridge.cv2_to_imgmsg(result_img, encoding="rgb8")
        self._pub_result_action_img.publish(pub_img_msg)
        # time.sleep(1 / self.img_topic_hz)

    def pixelTo3D(self, key_point: np.ndarray, confidence, cv_d: np.ndarray) -> Dict[Point, float]:
        """
        (x, y)で表される2次元上の画像座標を3次元座標に変換する関数
        Args:
            key_point(np.ndarray): キーポイントの画像上の2点 [x, y],
            cv_d(np.ndarray): depth image
        Returns:
            Tuple (geometry_msgs.msg.Point型の座標情報, 信頼値)
            計算に失敗した場合はFalseを返す
        """

        point_x, point_y = key_point[0], key_point[1]
        # confidence = key_point[2]
        self.get_logger().debug("point_x: " + str(point_x))
        self.get_logger().debug("point_y: " + str(point_y))
        self.get_logger().debug("confidence: " + str(confidence))

        # キーポイントが入っていない場合
        if point_x == 0 and point_y == 0:
            self.get_logger().debug("x and y is 0")
            return False

        # 重心算出
        cx = int(point_x)
        cy = int(point_y)

        # 重心周辺のDepth取得（ノイズ対策）
        kernel = [-1, 0, 1]
        depth_list = []
        for y in kernel:
            for x in kernel:
                try:
                    depth = cv_d[cy + y, cx + x] * 0.001  # mmからmに変更
                    if depth > 0:
                        depth_list.append(depth)
                except Exception as e:
                    self.get_logger().debug(f"{e}")
                    continue

        if len(depth_list) != 0:
            # 座標算出
            uv = list(self.camera_model.projectPixelTo3dRay((cx, cy)))
            uv[:] = [x / uv[2] for x in uv]
            uv[:] = [x * np.mean(depth_list) for x in uv]
        else:
            self.get_logger().debug("depth_list length is 0")
            return False

        point = Point()
        point.x = uv[0] + self._p_offset_x
        point.y = uv[1] + self._p_offset_y
        point.z = uv[2] + self._p_offset_z

        return {"point": point, "conf": confidence}

    def create_pose_marker_array(self, pose_results, depth_img):
        people_keypoints_3d = []  # 複数人分のキーポイント
        array_msg_pose_3d = []  # publish用のデータはあとで作成するため，配列に一時保存する

        for poses in pose_results:
            self.get_logger().debug("キーポイントごとの3次元座標を算出する")
            key_points = poses.pred_instances.keypoints[0]
            keypoints_3d = []
            msg_pose_3d = Ax3DPose()  # 1人分の3次元キーポイント座標
            for id, key_point in enumerate(key_points):
                # 3次元座標算出
                confidence = key_points = poses.pred_instances.keypoint_scores[0][id]
                keypoint_3d = self.pixelTo3D(key_point, confidence, depth_img.copy())
                keypoints_3d.append(keypoint_3d)

                # rosにpublishするようのメッセージを作成
                msg_keypoint_3d = AxKeyPoint()  # 3次元座標と信頼値を入れる

                # 3次元座標が算出されたとき
                if keypoint_3d:
                    msg_keypoint_3d.point = keypoint_3d["point"]
                    msg_keypoint_3d.score = float(keypoint_3d["conf"])
                else:
                    # 算出できなかったときは信頼値を-1にする
                    # msg_keypoint_3d.point = Point(0, 0, 0)
                    msg_keypoint_3d.score = -1.0

                # キーポイントの3次元座標を適切なメッセージの場所に格納
                if id == MM_ACTION_NOSE:
                    msg_pose_3d.nose = msg_keypoint_3d
                elif id == MM_ACTION_LEFT_EYE:
                    msg_pose_3d.left_eye = msg_keypoint_3d
                elif id == MM_ACTION_RIGHT_EYE:
                    msg_pose_3d.right_eye = msg_keypoint_3d
                elif id == MM_ACTION_LEFT_EAR:
                    msg_pose_3d.left_ear = msg_keypoint_3d
                elif id == MM_ACTION_RIGHT_EAR:
                    msg_pose_3d.right_ear = msg_keypoint_3d
                elif id == MM_ACTION_LEFT_SHOULDER:
                    msg_pose_3d.left_shoulder = msg_keypoint_3d
                elif id == MM_ACTION_RIGHT_SHOULDER:
                    msg_pose_3d.right_shoulder = msg_keypoint_3d
                elif id == MM_ACTION_LEFT_ELBOW:
                    msg_pose_3d.left_elbow = msg_keypoint_3d
                elif id == MM_ACTION_RIGHT_ELBOW:
                    msg_pose_3d.right_elbow = msg_keypoint_3d
                elif id == MM_ACTION_LEFT_WRIST:
                    msg_pose_3d.left_wrist = msg_keypoint_3d
                elif id == MM_ACTION_RIGHT_WRIST:
                    msg_pose_3d.right_wrist = msg_keypoint_3d
                elif id == MM_ACTION_LEFT_HIP:
                    msg_pose_3d.left_hip = msg_keypoint_3d
                elif id == MM_ACTION_RIGHT_HIP:
                    msg_pose_3d.right_hip = msg_keypoint_3d
                elif id == MM_ACTION_LEFT_KNEE:
                    msg_pose_3d.left_knee = msg_keypoint_3d
                elif id == MM_ACTION_RIGHT_KNEE:
                    msg_pose_3d.right_knee = msg_keypoint_3d
                elif id == MM_ACTION_LEFT_ANKLE:
                    msg_pose_3d.left_ankle = msg_keypoint_3d
                elif id == MM_ACTION_RIGHT_ANKLE:
                    msg_pose_3d.right_ankle = msg_keypoint_3d

            people_keypoints_3d.append(keypoints_3d)
            array_msg_pose_3d.append(msg_pose_3d)

        return array_msg_pose_3d, people_keypoints_3d

    def create_marker(self, frame_id, ns, marker_id, keypoint_3d, person, point1, point2) -> Marker:
        """
        2つのポイントから描画用のマーカーを作成する関数
        """

        # 信頼値情報から描画するかどうかを選択
        if keypoint_3d[self.mm_keypoint_info[point1]["id"]]["conf"] > self.pose_th and keypoint_3d[self.mm_keypoint_info[point2]["id"]]["conf"] > self.pose_th:
            color = self.mm_skeleton_info[ns]["color"]

            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = Time().to_msg()  # ROS2ではTime().to_msg()でタイムスタンプを設定
            marker.lifetime = Duration(seconds=0.5).to_msg()  # ROS2ではDurationの設定方法が変更
            marker.ns = ns
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.color.r = color[2] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[0] / 255.0
            marker.color.a = 1.0
            marker.scale.x = 0.03
            # marker.pose.orientation.w = 1.0
            marker.points.append(keypoint_3d[self.mm_keypoint_info[point1]["id"]]["point"])
            marker.points.append(keypoint_3d[self.mm_keypoint_info[point2]["id"]]["point"])

            return marker

        else:
            return False

    def display3DPose(self, people_keypoints_3d, tf_frame: str) -> MarkerArray:
        """
        複数人物のキーポイント情報をMarkerArrayを使って可視化する関数
        """
        marker_array = MarkerArray()
        for i, keypoints_3d in enumerate(people_keypoints_3d):
            if keypoints_3d is not None:
                for key, value in self.mm_skeleton_info.items():
                    if keypoints_3d[self.mm_keypoint_info[value["link"][0]]["id"]] and keypoints_3d[self.mm_keypoint_info[value["link"][1]]["id"]]:
                        marker = self.create_marker(
                            tf_frame,
                            key,
                            (i * 20) + value["id"],
                            keypoints_3d,
                            None,
                            value["link"][0],
                            value["link"][1]
                        )
                        if marker:
                            marker_array.markers.append(marker)
        return marker_array

    def pose_estimation_callback(self):
        # 各画像に対してhuman_detectionを行う
        # TODO(yano)同期を取る
        if (self.cv_img is not None) and (self.cv_depth is not None):
            image = self.cv_img.copy()
            depth = self.cv_depth.copy()
            self.cv_img = None
            self.cv_depth = None
        else:
            return

        human_detection_result = self.human_detection_inference(image)
        pose_results = inference_topdown(self.pose_estimator, image, human_detection_result)
        ax_3d_pose_array, people_keypoints_3d = self.create_pose_marker_array(pose_results=pose_results, depth_img=depth)
        # 可視化用のマーカを配信
        marker_array = self.display3DPose(people_keypoints_3d, self.camera_tf_frame)
        self._pub_result_pose_marker.publish(marker_array)

        # pose messagesを送信
        ax_3d_pose_msg_array = Ax3DPoseWithLabelArray()
        for idx, ax_3d_pose in enumerate(ax_3d_pose_array):
            ax_3d_pose_msg = Ax3DPoseWithLabel()
            ax_3d_pose_msg.keypoints = ax_3d_pose
            ax_3d_pose_msg_array.people.append(ax_3d_pose_msg)

            try:
                bbox = human_detection_result[idx]
                ax_3d_pose_msg.x = int(bbox[0])
                ax_3d_pose_msg.y = int(bbox[1])
                ax_3d_pose_msg.w = int(bbox[2]) - int(bbox[0])
                ax_3d_pose_msg.h = int(bbox[3]) - int(bbox[1])
            except IndexError as e:
                self.get_logger().warning(f"Human undetected!! {e}")

        # imageを追加
        ax_3d_pose_msg_array.rgb = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
        # ax_3d_pose_msg_array.depth = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")  # TODO(yano) Depthも送る

        self._pub_ax_3d_poses.publish(ax_3d_pose_msg_array)

        data_samples = merge_data_samples(pose_results)
        self._pose_visualizer(image, data_samples)
        pose_estimator_result_cv = self.visualizer.get_image()
        pose_result_cv_msg = self.bridge.cv2_to_imgmsg(pose_estimator_result_cv, encoding="rgb8")
        self._pub_result_pose_img.publish(pose_result_cv_msg)

        # 推論結果などを配列に保存
        if self.use_action_recognition:
            self.human_detections.append(human_detection_result)
            self.human_poses.append(pose_results)
            self.original_frames.append(image)
            self.depth_frames.append(depth)

    def action_recognition_callback(self):
        """
        timerコールバック関数
        """
        try:
            # 一定フレーム分の認識結果がたまったときに実行
            if len(self.original_frames) == self.step_size:
                # action認識を行う
                frames = [mmcv.imresize(img, (self.new_w, self.new_h)) for img in self.original_frames]
                self.inference(frames=frames, human_detections=self.human_detections)

                # 配列の長さに整合性が取れなくなった場合は強制的なリセット
                # if len(self.original_frames) != len(self.human_detections):
                self.original_frames = []
                self.human_detections = []
            elif len(self.original_frames) > self.step_size:
                diff = len(self.original_frames) - self.step_size
                for _ in range(diff):
                    _ = self.original_frames.pop(0)
                    _ = self.human_detections.pop(0)

                frames = [mmcv.imresize(img, (self.new_w, self.new_h)) for img in self.original_frames]
                self.inference(frames=frames, human_detections=self.human_detections)

        except IndexError as e:
            self.original_frames = []
            self.human_detections = []
            self.get_logger().warn(f"{e}")


def main(args=None):
    rclpy.init(args=args)
    node = MmActionDetector()
    # rclpy.spin(node)
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
