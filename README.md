# MMLabが提供するパッケージのロスラッパー

- 最終検証日 2023/09/02

```bash
git clone --recursive git.mm_pkg
```

## 環境構築

- singularity build

```bash
cd mm_ws/env/
singularity build --sandbox --fakeroot sandbox_mm mm_environment.def
```

- singularity shell

```bash
singularity shell --nv -B /run/user/1000,/var/lib/dbus/machine-id sandbox_mm
source /entrypoint.sh
```

- 確認方法

```bash
cd mm_pkg/include/mmaction2/mmaction/utils/
python collect_env.py
```

- [テスト時の実行環境](./test_environment.md)

## mmdeploy

- モデルの量子化を行うパッケージ
  - mmactionは処理が重たいため，`onnx`形式などに量子化しなければリアルタイム動作は不可能

### 量子化までの手順

#### spatial-temporal detection

- 必要なファイルのダウンロード
  - [vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py](https://github.com/open-mmlab/mmaction2/blob/main/configs/detection/videomae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py)を使用

```bash
cd your/path/mmaction2
mim download mmaction2 --config vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb --dest .
```

- deploy
  - detectionのdeplpoymentは，mmdeployではなく`mmaction2`内のdeploymentから実行する必要がある

```bash
python3 tools/deployment/export_onnx_stdet.py vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-3dafab75.pth --num_frames 16 --shape 480 640 --output_file videomae.onnx
```

- 推論

```bash
python3 demo/demo_spatiotemporal_det_onnx.py demo/demo.mp4 demo/demo_spatiotemporal_det.mp4 --config ./vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py --onnx-file ./videomae.onnx --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth --action-score-thr 0.5 --label-map tools/data/ava/label_map.txt 
```

- onnx関連でエラーが出た場合
  - サイズがあっていないというエラー
  - （一旦の解決策）`demo/demo_spatiotemporal_det_onnx.py`の249行目のあとに以下を追記
    - `clip_len = 8`
    - ここの = のあとの数字は，エラーにあった `index: 2 Got: x Expected: y`のyの部分

```bash
onnxruntime.capi.onnxruntime_pybind11_state.InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Got invalid dimensions for input: input_tensor for the following indices
 index: 2 Got: 32 Expected: 8
 Please fix either the inputs or the model.

```

#### recogntiion

- **すべて仮想環境の中で実行すること!!**

- 必要なファイルのダウンロード
  - `--config`以降の部分は，使用したいモデルのconfigファイルの名前を入れる
  - ~~今回は，`slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py`を使用する． [[config]](https://github.com/open-mmlab/mmaction2/blob/main/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py)~~ 動作しなかった．
  - [vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py)を使用？

```bash
cd your/path/mmdeploy
mim download mmaction2 --config vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400 --dest .
```

- `mim download`がうまく行かない場合の対処法 [issues](./issues.md#mim_downloadができない)

- deploy

- 例1: `videomae`
  - 自分の環境ではdeviceを指定することはできなかった．原因究明中

```bash
python3 tools/deploy.py \
configs/mmaction/video-recognition/video-recognition_onnxruntime_static.py \ 
vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py \
vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth \
tests/data/arm_wrestling.mp4 \
--work-dir mmdeploy_models/mmaction/videomae/ort \
--show \
--dump-info
```

- 例2: `tsn`

```bash
cd your/path/mmdeploy
python3 tools/deploy.py \
configs/mmaction/video-recognition/video-recognition_onnxruntime_static.py \
tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py \
tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth \
tests/data/arm_wrestling.mp4 \
--work-dir mmdeploy_models/mmaction/tsn/ort \
--device cuda:0 \
--show \
--dump-info
```

## 推論

- not work

```bash
python3 demo/demo_spatiotemporal_det_onnx.py demo/demo.mp4 demo/demo_spatiotemporal_det.mp4 --config ../mmdeploy/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py --onnx-file ../mmdeploy/mmdeploy_models/mmaction/videomae/ort/end2end.onnx --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth --action-score-thr 0.5 --label-map tools/data/ava/label_map.txt 
```

