import os
import glob
from setuptools import find_packages, setup

package_name = 'mmlab_ros_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # モデルとconfigの追加はここに書く
        (os.path.join('share', package_name, 'config/mmaction/detection/video_mae/'), glob.glob('config/mmaction/detection/video_mae/*.py')),
        (os.path.join('share', package_name, 'config/mmaction/label_map/'), glob.glob('config/mmaction/label_map/*.txt')),
        (os.path.join('share', package_name, 'config/mmaction/detection/recognition/tsn/ort/'), glob.glob('config/mmaction/recognition/tsn/ort/*.json')),
        (os.path.join('share', package_name, 'config/mmaction/detection/recognition/videomae/ort'), glob.glob('config/mmaction/recognition//videomae/ort/*.json')),
        (os.path.join('share', package_name, 'config/mmdetection/faster_rcnn/'), glob.glob('config/mmdetection/faster_rcnn/*.py')),
        (os.path.join('share', package_name, 'config/mmpose/seresnet/'), glob.glob('config/mmpose/seresnet/*.py')),

        # Tracking
        (os.path.join('share', package_name, 'config/mmtracking/reid/'), glob.glob('config/mmtracking/reid/*.py')),
        (os.path.join('share', package_name, 'config/mmtracking/_base_/'), glob.glob('config/mmtracking/_base_/*.py')),
        (os.path.join('share', package_name, 'config/mmtracking/_base_/datasets/'), glob.glob('config/mmtracking/_base_/datasets/*.py')),

        # models
        (os.path.join('share', package_name, 'config/ckpt/'), glob.glob('config/ckpt/*.pth')),
        (os.path.join('share', package_name, 'config/ckpt/'), glob.glob('config/ckpt/*.onnx')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='toyota',
    maintainer_email='fryuzoo@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "mmaction_spation_det = mmlab_ros_node.mmaction_spatio_det_onnx:main",
        ],
    },
)
