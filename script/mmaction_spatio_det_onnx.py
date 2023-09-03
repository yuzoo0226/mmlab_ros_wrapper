#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import mmcv
import time
import copy
import rospy
import torch
import queue
import atexit
import roslib
import logging
import mmengine
import threading
import numpy as np
import onnxruntime
from typing import List
from abc import ABCMeta, abstractmethod

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# class BaseHumanDetector(metaclass=ABCMeta):
#     """Base class for Human Dector.

#     Args:
#         device (str): CPU/CUDA device option.
#     """

#     def __init__(self, device):
#         self.device = torch.device(device)

#     @abstractmethod
#     def _do_detect(self, image):
#         """Get human bboxes with shape [n, 4].

#         The format of bboxes is (xmin, ymin, xmax, ymax) in pixels.
#         """

#     def predict(self, task):
#         """Add keyframe bboxes to task."""
#         # keyframe idx == (clip_len * frame_interval) // 2
#         keyframe = task.frames[len(task.frames) // 2]

#         # call detector
#         bboxes = self._do_detect(keyframe)

#         # convert bboxes to torch.Tensor and move to target device
#         if isinstance(bboxes, np.ndarray):
#             bboxes = torch.from_numpy(bboxes).to(self.device)
#         elif isinstance(bboxes, torch.Tensor) and bboxes.device != self.device:
#             bboxes = bboxes.to(self.device)

#         # update task
#         task.add_bboxes(bboxes)

#         return task


# class MmdetHumanDetector(BaseHumanDetector):
#     """Wrapper for mmdetection human detector.

#     Args:
#         config (str): Path to mmdetection config.
#         ckpt (str): Path to mmdetection checkpoint.
#         device (str): CPU/CUDA device option.
#         score_thr (float): The threshold of human detection score.
#         person_classid (int): Choose class from detection results.
#             Default: 0. Suitable for COCO pretrained models.
#     """

#     def __init__(self, config, ckpt, device, score_thr, person_classid=0):
#         super().__init__(device)
#         self.model = init_detector(config, ckpt, device=device)
#         self.person_classid = person_classid
#         self.score_thr = score_thr

#     def _do_detect(self, image):
#         """Get bboxes in shape [n, 4] and values in pixels."""
#         det_data_sample = inference_detector(self.model, image)
#         pred_instance = det_data_sample.pred_instances.cpu().numpy()
#         # We only keep human detection bboxs with score larger
#         # than `det_score_thr` and category id equal to `det_cat_id`.
#         valid_idx = np.logical_and(pred_instance.labels == self.person_classid,
#                                    pred_instance.scores > self.score_thr)
#         bboxes = pred_instance.bboxes[valid_idx]
#         # result = result[result[:, 4] >= self.score_thr][:, :4]
#         return bboxes


# # MMAction2 demo/webcam_demo_spatiotemporal_det.pyより
# class TaskInfo:
#     """Wapper for a clip.

#     Transmit data around three threads.

#     1) Read Thread: Create task and put task into read queue. Init `frames`,
#         `processed_frames`, `img_shape`, `ratio`, `clip_vis_length`.
#     2) Main Thread: Get data from read queue, predict human bboxes and stdet
#         action labels, draw predictions and put task into display queue. Init
#         `display_bboxes`, `stdet_bboxes` and `action_preds`, update `frames`.
#     3) Display Thread: Get data from display queue, show/write frames and
#         delete task.
#     """

#     def __init__(self):
#         self.id = -1

#         # raw frames, used as human detector input, draw predictions input
#         # and output, display input
#         self.frames = None

#         # stdet params
#         self.processed_frames = None  # model inputs
#         self.frames_inds = None  # select frames from processed frames
#         self.img_shape = None  # model inputs, processed frame shape
#         # `action_preds` is `list[list[tuple]]`. The outer brackets indicate
#         # different bboxes and the intter brackets indicate different action
#         # results for the same bbox. tuple contains `class_name` and `score`.
#         self.action_preds = None  # stdet results

#         # human bboxes with the format (xmin, ymin, xmax, ymax)
#         self.display_bboxes = None  # bboxes coords for self.frames
#         self.stdet_bboxes = None  # bboxes coords for self.processed_frames
#         self.ratio = None  # processed_frames.shape[1::-1]/frames.shape[1::-1]

#         # for each clip, draw predictions on clip_vis_length frames
#         self.clip_vis_length = -1

#     def add_frames(self, idx, frames, processed_frames):
#         """Add the clip and corresponding id.

#         Args:
#             idx (int): the current index of the clip.
#             frames (list[ndarray]): list of images in "BGR" format.
#             processed_frames (list[ndarray]): list of resize and normed images
#                 in "BGR" format.
#         """
#         self.frames = frames
#         self.processed_frames = processed_frames
#         self.id = idx
#         self.img_shape = processed_frames[0].shape[:2]

#     def add_bboxes(self, display_bboxes):
#         """Add correspondding bounding boxes."""
#         self.display_bboxes = display_bboxes
#         self.stdet_bboxes = display_bboxes.clone()
#         self.stdet_bboxes[:, ::2] = self.stdet_bboxes[:, ::2] * self.ratio[0]
#         self.stdet_bboxes[:, 1::2] = self.stdet_bboxes[:, 1::2] * self.ratio[1]

#     def add_action_preds(self, preds):
#         """Add the corresponding action predictions."""
#         self.action_preds = preds

#     def get_model_inputs(self, device):
#         """Convert preprocessed images to MMAction2 STDet model inputs."""
#         cur_frames = [self.processed_frames[idx] for idx in self.frames_inds]
#         input_array = np.stack(cur_frames).transpose((3, 0, 1, 2))[np.newaxis]
#         input_tensor = torch.from_numpy(input_array).to(device)
#         datasample = ActionDataSample()
#         datasample.proposals = InstanceData(bboxes=self.stdet_bboxes)
#         datasample.set_metainfo(dict(img_shape=self.img_shape))

#         return dict(
#             inputs=input_tensor, data_samples=[datasample], mode='predict')


# class ClipHelper:
#     """Multithrading utils to manage the lifecycle of task."""

#     def __init__(self,
#                  config,
#                  display_height=0,
#                  display_width=0,
#                  input_video=0,
#                  predict_stepsize=40,
#                  output_fps=25,
#                  clip_vis_length=8,
#                  out_filename=None,
#                  show=True,
#                  stdet_input_shortside=256):
#         # stdet sampling strategy
#         val_pipeline = config.val_pipeline
#         sampler = [x for x in val_pipeline
#                    if x['type'] == 'SampleAVAFrames'][0]
#         clip_len, frame_interval = sampler['clip_len'], sampler[
#             'frame_interval']
#         self.window_size = clip_len * frame_interval

#         # asserts
#         assert (out_filename or show), \
#             'out_filename and show cannot both be None'
#         assert clip_len % 2 == 0, 'We would like to have an even clip_len'
#         assert clip_vis_length <= predict_stepsize
#         assert 0 < predict_stepsize <= self.window_size

#         # source params
#         try:
#             self.cap = cv2.VideoCapture(int(input_video))
#             self.webcam = True
#         except ValueError:
#             self.cap = cv2.VideoCapture(input_video)
#             self.webcam = False
#         assert self.cap.isOpened()

#         # stdet input preprocessing params
#         h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.stdet_input_size = mmcv.rescale_size(
#             (w, h), (stdet_input_shortside, np.Inf))
#         img_norm_cfg = dict(
#             mean=np.array(config.model.data_preprocessor.mean),
#             std=np.array(config.model.data_preprocessor.std),
#             to_rgb=False)
#         self.img_norm_cfg = img_norm_cfg

#         # task init params
#         self.clip_vis_length = clip_vis_length
#         self.predict_stepsize = predict_stepsize
#         self.buffer_size = self.window_size - self.predict_stepsize
#         frame_start = self.window_size // 2 - (clip_len // 2) * frame_interval
#         self.frames_inds = [
#             frame_start + frame_interval * i for i in range(clip_len)
#         ]
#         self.buffer = []
#         self.processed_buffer = []

#         # output/display params
#         if display_height > 0 and display_width > 0:
#             self.display_size = (display_width, display_height)
#         elif display_height > 0 or display_width > 0:
#             self.display_size = mmcv.rescale_size(
#                 (w, h), (np.Inf, max(display_height, display_width)))
#         else:
#             self.display_size = (w, h)
#         self.ratio = tuple(
#             n / o for n, o in zip(self.stdet_input_size, self.display_size))
#         if output_fps <= 0:
#             self.output_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
#         else:
#             self.output_fps = output_fps
#         self.show = show
#         self.video_writer = None
#         if out_filename is not None:
#             self.video_writer = self.get_output_video_writer(out_filename)
#         display_start_idx = self.window_size // 2 - self.predict_stepsize // 2
#         self.display_inds = [
#             display_start_idx + i for i in range(self.predict_stepsize)
#         ]

#         # display multi-theading params
#         self.display_id = -1  # task.id for display queue
#         self.display_queue = {}
#         self.display_lock = threading.Lock()
#         self.output_lock = threading.Lock()

#         # read multi-theading params
#         self.read_id = -1  # task.id for read queue
#         self.read_id_lock = threading.Lock()
#         self.read_queue = queue.Queue()
#         self.read_lock = threading.Lock()
#         self.not_end = True  # cap.read() flag

#         # program state
#         self.stopped = False

#         atexit.register(self.clean)

#     def read_fn(self):
#         """Main function for read thread.

#         Contains three steps:

#         1) Read and preprocess (resize + norm) frames from source.
#         2) Create task by frames from previous step and buffer.
#         3) Put task into read queue.
#         """
#         was_read = True
#         start_time = time.time()
#         while was_read and not self.stopped:
#             # init task
#             task = TaskInfo()
#             task.clip_vis_length = self.clip_vis_length
#             task.frames_inds = self.frames_inds
#             task.ratio = self.ratio

#             # read buffer
#             frames = []
#             processed_frames = []
#             if len(self.buffer) != 0:
#                 frames = self.buffer
#             if len(self.processed_buffer) != 0:
#                 processed_frames = self.processed_buffer

#             # read and preprocess frames from source and update task
#             with self.read_lock:
#                 before_read = time.time()
#                 read_frame_cnt = self.window_size - len(frames)
#                 while was_read and len(frames) < self.window_size:
#                     was_read, frame = self.cap.read()
#                     if not self.webcam:
#                         # Reading frames too fast may lead to unexpected
#                         # performance degradation. If you have enough
#                         # resource, this line could be commented.
#                         time.sleep(1 / self.output_fps)
#                     if was_read:
#                         frames.append(mmcv.imresize(frame, self.display_size))
#                         processed_frame = mmcv.imresize(
#                             frame, self.stdet_input_size).astype(np.float32)
#                         _ = mmcv.imnormalize_(processed_frame,
#                                               **self.img_norm_cfg)
#                         processed_frames.append(processed_frame)
#             task.add_frames(self.read_id + 1, frames, processed_frames)

#             # update buffer
#             if was_read:
#                 self.buffer = frames[-self.buffer_size:]
#                 self.processed_buffer = processed_frames[-self.buffer_size:]

#             # update read state
#             with self.read_id_lock:
#                 self.read_id += 1
#                 self.not_end = was_read

#             self.read_queue.put((was_read, copy.deepcopy(task)))
#             cur_time = time.time()
#             logger.debug(
#                 f'Read thread: {1000*(cur_time - start_time):.0f} ms, '
#                 f'{read_frame_cnt / (cur_time - before_read):.0f} fps')
#             start_time = cur_time

#     def display_fn(self):
#         """Main function for display thread.

#         Read data from display queue and display predictions.
#         """
#         start_time = time.time()
#         while not self.stopped:
#             # get the state of the read thread
#             with self.read_id_lock:
#                 read_id = self.read_id
#                 not_end = self.not_end

#             with self.display_lock:
#                 # If video ended and we have display all frames.
#                 if not not_end and self.display_id == read_id:
#                     break

#                 # If the next task are not available, wait.
#                 if (len(self.display_queue) == 0 or
#                         self.display_queue.get(self.display_id + 1) is None):
#                     time.sleep(0.02)
#                     continue

#                 # get display data and update state
#                 self.display_id += 1
#                 was_read, task = self.display_queue[self.display_id]
#                 del self.display_queue[self.display_id]
#                 display_id = self.display_id

#             # do display predictions
#             with self.output_lock:
#                 if was_read and task.id == 0:
#                     # the first task
#                     cur_display_inds = range(self.display_inds[-1] + 1)
#                 elif not was_read:
#                     # the last task
#                     cur_display_inds = range(self.display_inds[0],
#                                              len(task.frames))
#                 else:
#                     cur_display_inds = self.display_inds

#                 for frame_id in cur_display_inds:
#                     frame = task.frames[frame_id]
#                     if self.show:
#                         cv2.imshow('Demo', frame)
#                         cv2.waitKey(int(1000 / self.output_fps))
#                     if self.video_writer:
#                         self.video_writer.write(frame)

#             cur_time = time.time()
#             logger.debug(
#                 f'Display thread: {1000*(cur_time - start_time):.0f} ms, '
#                 f'read id {read_id}, display id {display_id}')
#             start_time = cur_time

#     def __iter__(self):
#         return self

#     def __next__(self):
#         """Get data from read queue.

#         This function is part of the main thread.
#         """
#         if self.read_queue.qsize() == 0:
#             time.sleep(0.02)
#             return not self.stopped, None

#         was_read, task = self.read_queue.get()
#         if not was_read:
#             # If we reach the end of the video, there aren't enough frames
#             # in the task.processed_frames, so no need to model inference
#             # and draw predictions. Put task into display queue.
#             with self.read_id_lock:
#                 read_id = self.read_id
#             with self.display_lock:
#                 self.display_queue[read_id] = was_read, copy.deepcopy(task)

#             # main thread doesn't need to handle this task again
#             task = None
#         return was_read, task

#     def start(self):
#         """Start read thread and display thread."""
#         self.read_thread = threading.Thread(
#             target=self.read_fn, args=(), name='VidRead-Thread', daemon=True)
#         self.read_thread.start()
#         self.display_thread = threading.Thread(
#             target=self.display_fn,
#             args=(),
#             name='VidDisplay-Thread',
#             daemon=True)
#         self.display_thread.start()

#         return self

#     def clean(self):
#         """Close all threads and release all resources."""
#         self.stopped = True
#         self.read_lock.acquire()
#         self.cap.release()
#         self.read_lock.release()
#         self.output_lock.acquire()
#         cv2.destroyAllWindows()
#         if self.video_writer:
#             self.video_writer.release()
#         self.output_lock.release()

#     def join(self):
#         """Waiting for the finalization of read and display thread."""
#         self.read_thread.join()
#         self.display_thread.join()

#     def display(self, task):
#         """Add the visualized task to the display queue.

#         Args:
#             task (TaskInfo object): task object that contain the necessary
#             information for prediction visualization.
#         """
#         with self.display_lock:
#             self.display_queue[task.id] = (True, task)

#     def get_output_video_writer(self, path):
#         """Return a video writer object.

#         Args:
#             path (str): path to the output video file.
#         """
#         return cv2.VideoWriter(
#             filename=path,
#             fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
#             fps=float(self.output_fps),
#             frameSize=self.display_size,
#             isColor=True)


# 自作クラス
class MmActionDetector():
    def __init__(self) -> None:
        self.pkg_dir = roslib.packages.get_pkg_dir("mmlab_ros_wrapper")
        print(f"your package dir is: {self.pkg_dir}")

        ############################################
        # ロボットに依存するパラメータ
        ############################################
        self.img_width = 640
        self.img_height = 480

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
        self.label_map_path = "configs/mmaction/label_map/ava_label_map.txt"
        self.label_map_path = os.path.join(self.pkg_dir, self.label_map_path)
        print(f"your label map path is {self.label_map_path}")
        self.label_map = self.load_label_map(self.label_map_path)

        # mmaction2モデルパス
        self.action_onnx_path = "configs/mmaction/detection/video_mae/videomae.onnx"
        self.action_config = "configs/mmaction/detection/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py"

        self.action_onnx_path = os.path.join(self.pkg_dir, self.action_onnx_path)
        self.action_config = os.path.join(self.pkg_dir, self.action_config)
        self.session = onnxruntime.InferenceSession(self.action_onnx_path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

        # detection モデルパス
        self.detection_pth_path = "configs/mmdetection/faster_rcnn/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth"
        self.detection_config = "configs/mmdetection/faster_rcnn/faster-rcnn_r50_fpn_2x_coco_infer.py"

        self.detection_pth_path = os.path.join(self.pkg_dir, self.detection_pth_path)
        self.detection_config = os.path.join(self.pkg_dir, self.detection_config)
        self.detection_model = init_detector(self.detection_config, self.detection_pth_path, self.device)

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
        self.img_topic_name = "/camera/rgb/image_raw"
        rospy.Subscriber(self.img_topic_name, Image, callback=self.run, queue_size=1)
        self.result_pub = rospy.Publisher("/mm_wrapper/action/result/image", Image, queue_size=10)

        # プログラムに必要な変数
        self.original_frames = []
        self.human_detections = []
        self.frame_counter = 0

        self.text_fontface = cv2.FONT_HERSHEY_DUPLEX
        self.text_fontscale = 0.5
        self.text_fontcolor = (255, 255, 255)
        self.text_thickness = 1
        self.text_linetype = 1
        plate='03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
        plate = plate.split('-')
        self.plate = [self.hex2color(h) for h in plate]

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

    def inference(self, frames: List[np.ndarray], human_detections):
        """
        actionの推論を行う関数
        frames(List[np.ndarray]): 画像が複数枚入っている配列
        """

        for i in range(len(human_detections)):
            det = human_detections[i].tolist()
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

        result_img = self.draw_one_image(frame=frames[-1], bboxes=det, preds=prediction)
        pub_img_msg = self.bridge.cv2_to_imgmsg(result_img, encoding="rgb8")
        self.result_pub.publish(pub_img_msg)

    def run(self, msg: Image):
        """
        imgトピックを受信したときのコールバック関数
        msg(Image): 推論対象とする image msg
        """
        rospy.logdebug("get rgb image")
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # 各画像に対してhuman_detectionを行う
        human_detection_result = self.human_detection_inference(image)
        # 推論結果などを配列に保存
        try:
            self.human_detections.append(human_detection_result[0])
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
            rospy.logwarn(e)


if __name__ == "__main__":
    rospy.init_node("mmaction2_server")
    cls = MmActionDetector()

    while not rospy.is_shutdown():
        rospy.sleep(0.1)
