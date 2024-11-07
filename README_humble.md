## とりあえずの対応

# mmactionのonnx化

```bash
cd include/mm_pkg
git clone https://github.com/open-mmlab/mmaction2.git
# commit id is 4d6c934
```

```bash
python3 tools/deployment/export_onnx_stdet.py ../../../mmlab_ros_node/config/mmaction/detection/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py  ../../../mmlab_ros_node/config/ckpt/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth  --num_frames 16 --output_file videomae.onnx
```
