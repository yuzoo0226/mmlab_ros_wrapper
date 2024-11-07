#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import mmcv
import time
import rclpy

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
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from rclpy.node import Node


class MmActionDetector(Node):
    def __init__(self) -> None:
        super().__init__("mmaction_node")
        self.package_dir = get_package_share_directory("mmlab_ros_node")
        print(f"your shared package dir is: {self.package_dir}")

        ############################################
        # ロボットに依存するパラメータ
        ############################################
        self.img_width = 640
        self.img_height = 480
        self.img_topic_hz = 30

        ############################################
        # MMAction2固有のパラメータ
        ############################################
        self.short_side = 256
        self.predict_stepsize = 8
        self.step_size = 16  # xフレームに一回の推論
        self.max_labels_per_bbox = 5

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
        print(f"your label map path is {self.label_map_path}")
        self.label_map = self.load_label_map(self.label_map_path)

        # detection モデルパス
        self.detection_pth_path = "config/ckpt/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
        self.detection_config = "config/mmdetection/faster_rcnn/faster-rcnn_r50_fpn_2x_coco_infer.py"

        self.detection_pth_path = os.path.join(self.package_dir, self.detection_pth_path)
        self.detection_config = os.path.join(self.package_dir, self.detection_config)
        self.detection_model = init_detector(self.detection_config, self.detection_pth_path, self.device)

        # mmaction2モデルパス
        self.action_onnx_path = "config/ckpt/videomae.onnx"
        self.action_config = "config/mmaction/detection/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py"

        self.action_onnx_path = os.path.join(self.package_dir, self.action_onnx_path)
        self.action_config = os.path.join(self.package_dir, self.action_config)
        self.session = onnxruntime.InferenceSession(self.action_onnx_path)

        # self.display_width = 700
        # self.display_height = 500
        # self.show = False

        ############################################
        # human detectorの初期化
        ############################################
        # init clip helper
        # config = Config.fromfile(self.action_config)

        # self.clip_helper = ClipHelper(
        #     config=config,
        #     display_height=self.display_height,
        #     display_width=self.display_width,
        #     out_filename="./temp,mp4",
        #     show=self.show
        # )

        # self.human_detector = MmdetHumanDetector(self.detection_config, self.detection_pth_path, self.device, self.det_score_thr)

        # # Get clip_len, frame_interval and calculate center index of each clip
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
        # self.img_topic_name = "/camera/rgb/image_raw"

        custom_qos_profile = QoSProfile(depth=1, reliability=qos_profile_sensor_data.reliability, durability=qos_profile_sensor_data.durability, history=qos_profile_sensor_data.history)

        self._rgb_msg = None
        self._depth_msg = None
        self.cv_img = None
        self._rgb_sub = self.create_subscription(Image, "/camera/color/image_raw", self.rgb_callback, custom_qos_profile)
        self._pub_result_img = self.create_publisher(Image, "~/result_image", qos_profile_sensor_data)

        self.timer = self.create_timer(0.05, self.run)

        # プログラムに必要な変数
        self.original_frames = []
        self.human_detections = []
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
        self.cv_img = self.bridge.imgmsg_to_cv2(msg)

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
            print(human_detections[i][0])
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
        print(prediction)

        result_imgs = self.draw_all_images(frames=frames, bboxes=det, preds=prediction)
        result_img = result_imgs[-1]
        # for result_img in result_imgs:
        pub_img_msg = self.bridge.cv2_to_imgmsg(result_img, encoding="rgb8")
        self._pub_result_img.publish(pub_img_msg)
        # time.sleep(1 / self.img_topic_hz)

    def run(self):
        """
        timerコールバック関数
        """
        # 各画像に対してhuman_detectionを行う
        if self.cv_img is not None:
            image = self.cv_img
        else:
            return

        human_detection_result = self.human_detection_inference(image)

        # 推論結果などを配列に保存
        try:
            self.human_detections.append(human_detection_result)
            self.original_frames.append(image)

            # 一定フレーム分の認識結果がたまったときに実行
            if len(self.original_frames) >= self.step_size:
                # action認識を行う
                frames = [mmcv.imresize(img, (self.new_w, self.new_h)) for img in self.original_frames]
                self.inference(frames=frames, human_detections=self.human_detections)

                # 配列の長さに整合性が取れなくなった場合は強制的なリセット
                # if len(self.original_frames) != len(self.human_detections):
                self.original_frames = []
                self.human_detections = []

        except IndexError as e:
            self.original_frames = []
            self.human_detections = []
            self.get_logger().warn(f"{e}")

        self.cv_img = None


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
