#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import mmcv
import time
import rclpy
import copy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from ament_index_python.packages import get_package_share_directory
from coordinate_transform_util_ros.transform import TransformUtils
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, Point, Quaternion
from mmlab_ros_msgs.msg import Ax3DPose, AxKeyPoint, Ax3DPoseWithLabel, Ax3DPoseWithLabelArray
from std_msgs.msg import Header
from std_srvs.srv import SetBool, Trigger


class MMPoseCollisionRegister(Node):
    def __init__(self) -> None:
        super().__init__("mmpose_collision_register")
        self.package_dir = get_package_share_directory("mmlab_ros_node")
        self.get_logger().info(f"your shared package dir is: {self.package_dir}")
        self._transform_utils = TransformUtils.get_instance(self)

        self.camera_frame = "head_rgbd_sensor_rgb_frame"
        self.target_frame = "map"

        # ros interface
        self._sub_ax_3d_poses = self.create_subscription(Ax3DPoseWithLabelArray, "/mmaction_node/people_poses", self.human_pose_callback, qos_profile_sensor_data)
        self._pub_collision_object = self.create_publisher(CollisionObject, "/collision_environment_server/collision_object", 10)

        self.unregister_human_bbox_srv = self.create_service(Trigger, '~/unregister_human_collisions', self.unregister_human_bbox)
        self.bool_srv = self.create_service(SetBool, '~/run_enable', self.set_bool_callback)

        # timer callback
        self.pose_box_timer = self.create_timer(0.1, self.register_human_boxes)

        self.run_enable = True
        self.score_th = 0.6
        self.current_human_poses = []
        self.object_id = "mm_human_poses"

        self.msg_add = 0
        self.msg_remove = 1

    def human_pose_callback(self, msg: Ax3DPoseWithLabelArray) -> None:
        self.current_timestamp = msg.header.stamp
        self.current_human_poses = msg.people
        self.get_logger().debug("callback human poses")

    def unregister_human_bbox(self) -> bool:
        try:
            unregist_collision_human_msg = CollisionObject()
            unregist_collision_human_msg.header = Header()
            unregist_collision_human_msg.header.frame_id = self.target_frame
            unregist_collision_human_msg.operation = CollisionObject.REMOVE
            unregist_collision_human_msg.id = self.object_id
            self.get_logger().info("unregist human bboxes successfully")
            self._pub_collision_object.publish(unregist_collision_human_msg)
            return True

        except Exception as e:
            self.get_logger().warn(f"{e}")
            self.get_logger().warn("[failed] Cannot unregist human bboxes")
            return False

    def set_bool_callback(self, request, response):
        """run enableを制御
        """
        self.run_enable = request.data
        self.get_logger().info(f"mmpose_collision_server run_enable is: {self.run_enable}")

    def cb_unregister_human_boxes(self, request, response):
        # サービスリクエストを処理する部分
        status = self.unregister_human_bbox()
        response.success = status
        if status:
            response.message = "unregist human bboxes successfully"
        else:
            response.message = "Cannot unregist human bboxes"
        return response

    def register_human_boxes(self) -> None:
        """
        人の3次元BBoxを障害物として配信
        """
        if self.run_enable is False:
            return

        data = self.current_human_poses

        if data != []:
            collision_human_msg = CollisionObject()
            collision_human_msg.header = Header()
            collision_human_msg.header.frame_id = self.target_frame
            collision_human_msg.operation = CollisionObject.ADD
            collision_human_msg.id = self.object_id

            camera_to_map = self._transform_utils.get_pose(self.target_frame, self.camera_frame, self.current_timestamp)
            primitives_poses = []
            primitives_shapes = []

            for human_pose in data:
                check_points = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
                temp_human_points_x = []
                temp_human_points_y = []
                temp_human_points_z = []

                # すべてのkeypointから閾値以上のスコアのpointを抽出
                for point_name in check_points:
                    point = getattr(human_pose.keypoints, point_name).point
                    score = getattr(human_pose.keypoints, point_name).score

                    if score > self.score_th:

                        temp_human_pose = Pose()
                        temp_human_pose.position = point
                        temp_human_pose.orientation.x = 0.0
                        temp_human_pose.orientation.y = 0.0
                        temp_human_pose.orientation.z = 0.0
                        temp_human_pose.orientation.w = 1.0

                        temp_human_pose_map = self._transform_utils.transform_pose(camera_to_map, temp_human_pose)

                        # point を使って処理を行う
                        self.get_logger().debug(f"{point_name} position via {self.target_frame}: {temp_human_pose_map.position}")
                        temp_human_points_x.append(temp_human_pose_map.position.x)
                        temp_human_points_y.append(temp_human_pose_map.position.y)
                        temp_human_points_z.append(temp_human_pose_map.position.z)

                # TODO(yano) サイズのバリデーションを入れる

                # 各座標の最大値と最小値を計算
                if temp_human_points_x and temp_human_points_y and temp_human_points_z:
                    max_x = max(temp_human_points_x)
                    min_x = min(temp_human_points_x)
                    self.get_logger().debug(f"Max X: {max_x}, Min X: {min_x}")

                    max_y = max(temp_human_points_y)
                    min_y = min(temp_human_points_y)
                    self.get_logger().debug(f"Max Y: {max_y}, Min Y: {min_y}")

                    max_z = max(temp_human_points_z)
                    min_z = min(temp_human_points_z)
                    self.get_logger().debug(f"Max Z: {max_z}, Min Z: {min_z}")

                    pose = Pose()
                    pose.position.x = float((max_x + min_x) / 2.0)
                    pose.position.y = float((max_y + min_y) / 2.0)
                    pose.position.z = float((max_z + min_z) / 2.0)
                    pose.orientation.w = 1.0

                    size_x = float(abs(max_x - min_x))
                    size_y = float(abs(max_y - min_y))
                    size_z = float(abs(max_z - min_z))

                    shape = SolidPrimitive()
                    shape.type = SolidPrimitive.BOX
                    shape.dimensions = [size_x, size_y, size_z]

                    primitives_shapes.append(shape)
                    primitives_poses.append(pose)

            collision_human_msg.primitives = primitives_shapes
            collision_human_msg.primitive_poses = primitives_poses

            self._pub_collision_object.publish(collision_human_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MMPoseCollisionRegister()
    # rclpy.spin(node)
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
