# issues

- 環境構築中に起こったバグをまとめたもの

## mim_downloadができない

- エラー内容

```bash
 > mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb --dest .
Traceback (most recent call last):
  File "<frozen importlib._bootstrap>", line 602, in _exec
  File "<frozen zipimport>", line 259, in load_module
  File "/usr/share/python-wheels/pkg_resources-0.0.0-py2.py3-none-any.whl/pkg_resources/__init__.py", line 57, in <module>
ImportError: cannot import name 'six' from 'pkg_resources.extern' (/usr/local/lib/python3.8/dist-packages/pkg_resources/extern/__init__.py)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/bin/mim", line 8, in <module>
    sys.exit(cli())
  File "/usr/local/lib/python3.8/dist-packages/click/core.py", line 1157, in __call__
    return self.main(*args, **kwargs)
  File "/usr/local/lib/python3.8/dist-packages/click/core.py", line 1078, in main
    rv = self.invoke(ctx)
  File "/usr/local/lib/python3.8/dist-packages/click/core.py", line 1688, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "/usr/local/lib/python3.8/dist-packages/click/core.py", line 1434, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/usr/local/lib/python3.8/dist-packages/click/core.py", line 783, in invoke
    return __callback(*args, **kwargs)
  File "/usr/local/lib/python3.8/dist-packages/mim/commands/download.py", line 70, in cli
    download(package, configs, dest_root, check_certificate, dataset)
  File "/usr/local/lib/python3.8/dist-packages/mim/commands/download.py", line 107, in download
    return _download_configs(package, configs, dest_root,
  File "/usr/local/lib/python3.8/dist-packages/mim/commands/download.py", line 126, in _download_configs
    if not is_installed(package):
  File "/usr/local/lib/python3.8/dist-packages/mim/utils/utils.py", line 59, in is_installed
    importlib.reload(pkg_resources)
  File "/usr/lib/python3.8/importlib/__init__.py", line 169, in reload
    _bootstrap._exec(spec, module)
  File "<frozen importlib._bootstrap>", line 608, in _exec
KeyError: 'pkg_resources'

```

### how to fix

- mmcvのバージョンが高すぎると起きるバグ？
  - `pip install mmcv==1.7.1`を実行し，mmcvのバージョンを下げてから再度 `mim download ~~`を実行．
  - `mim download ~~`が終わったあとは，再度 `pip install mmcv==2.0.1`を実行してバージョンを戻す．


## same device

- use cuda and cpu same time

```bash
 > python3 include/mmaction2/tools/deployment/export_onnx_stdet.py ./configs/mm_action/detection/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py ./configs/mm_action/detection/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-3dafab75.pth --num_frames 16 --output_file videomae.onnx --device cuda:0
cuda:0
Loads checkpoint by local backend from path: ./configs/mm_action/detection/video_mae/vit-base-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb_20230314-3dafab75.pth
Model output shape: torch.Size([3, 81])
cuda:0
cuda:0
/usr/local/lib/python3.8/dist-packages/mmcv/cnn/bricks/wrappers.py:65: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
/usr/local/lib/python3.8/dist-packages/mmaction/models/backbones/vit_mae.py:362: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if (h, w) != self.grid_size:
/usr/local/lib/python3.8/dist-packages/mmcv/cnn/bricks/wrappers.py:167: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 5)):
/usr/local/lib/python3.8/dist-packages/mmcv/ops/roi_align.py:78: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'
/usr/local/lib/python3.8/dist-packages/mmcv/ops/roi_align.py:31: FutureWarning: 'torch.onnx._patch_torch._graph_op' is deprecated in version 1.13 and will be removed in version 1.14. Please note 'g.op()' is to be removed from torch.Graph. Please open a GitHub issue if you need this functionality..
  g.op('Constant', value_t=torch.tensor([0], dtype=torch.long)))
/usr/local/lib/python3.8/dist-packages/mmcv/ops/roi_align.py:26: FutureWarning: 'torch.onnx._patch_torch._graph_op' is deprecated in version 1.13 and will be removed in version 1.14. Please note 'g.op()' is to be removed from torch.Graph. Please open a GitHub issue if you need this functionality..
  return g.op('Gather', self, index, axis_i=dim)
/usr/local/lib/python3.8/dist-packages/mmcv/ops/roi_align.py:32: FutureWarning: 'torch.onnx._patch_torch._graph_op' is deprecated in version 1.13 and will be removed in version 1.14. Please note 'g.op()' is to be removed from torch.Graph. Please open a GitHub issue if you need this functionality..
  batch_indices = g.op('Squeeze', batch_indices, axes_i=[1])
/usr/local/lib/python3.8/dist-packages/mmcv/ops/roi_align.py:33: FutureWarning: 'torch.onnx._patch_torch._graph_op' is deprecated in version 1.13 and will be removed in version 1.14. Please note 'g.op()' is to be removed from torch.Graph. Please open a GitHub issue if you need this functionality..
  batch_indices = g.op(
/usr/local/lib/python3.8/dist-packages/mmcv/ops/roi_align.py:38: FutureWarning: 'torch.onnx._patch_torch._graph_op' is deprecated in version 1.13 and will be removed in version 1.14. Please note 'g.op()' is to be removed from torch.Graph. Please open a GitHub issue if you need this functionality..
  g.op(
/usr/local/lib/python3.8/dist-packages/mmcv/ops/roi_align.py:44: FutureWarning: 'torch.onnx._patch_torch._graph_op' is deprecated in version 1.13 and will be removed in version 1.14. Please note 'g.op()' is to be removed from torch.Graph. Please open a GitHub issue if you need this functionality..
  aligned_offset = g.op(
/usr/local/lib/python3.8/dist-packages/torch/onnx/symbolic_opset9.py:364: FutureWarning: 'torch.onnx._patch_torch._graph_op' is deprecated in version 1.13 and will be removed in version 1.14. Please note 'g.op()' is to be removed from torch.Graph. Please open a GitHub issue if you need this functionality..
  return g.op("Sub", self, other)
/usr/local/lib/python3.8/dist-packages/mmcv/ops/roi_align.py:50: FutureWarning: 'torch.onnx._patch_torch._graph_op' is deprecated in version 1.13 and will be removed in version 1.14. Please note 'g.op()' is to be removed from torch.Graph. Please open a GitHub issue if you need this functionality..
  return g.op(
/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py:687: UserWarning: Constant folding in symbolic shape inference fails: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument tensors in method wrapper_cat) (Triggered internally at ../torch/csrc/jit/passes/onnx/shape_type_inference.cpp:413.)
  _C._jit_pass_onnx_graph_shape_type_inference(
Traceback (most recent call last):
  File "include/mmaction2/tools/deployment/export_onnx_stdet.py", line 208, in <module>
    main()
  File "include/mmaction2/tools/deployment/export_onnx_stdet.py", line 167, in main
    torch.onnx.export(
  File "/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py", line 504, in export
    _export(
  File "/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py", line 1529, in _export
    graph, params_dict, torch_out = _model_to_graph(
  File "/usr/local/lib/python3.8/dist-packages/torch/onnx/utils.py", line 1172, in _model_to_graph
    params_dict = _C._jit_pass_onnx_constant_fold(
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument tensors in method wrapper_cat)
```


## onnx export

- no problem, export onnx file is complete!!
- you add `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`  when your inference time
  - `session = onnxruntime.InferenceSession(args.onnx_file, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])`

```bash
Successfully export the onnx file to videomae.onnx
Traceback (most recent call last):
  File "include/mmaction2/tools/deployment/export_onnx_stdet.py", line 207, in <module>
    main()
  File "include/mmaction2/tools/deployment/export_onnx_stdet.py", line 194, in main
    session = onnxruntime.InferenceSession(args.output_file)
  File "/usr/local/lib/python3.8/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 347, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "/usr/local/lib/python3.8/dist-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 375, in _create_inference_session
    raise ValueError(
ValueError: This ORT build has ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] enabled. Since ORT 1.9, you are required to explicitly set the providers parameter when instantiating InferenceSession. For example, onnxruntime.InferenceSession(..., providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'], ...)
```


## inference moviepy

- pip install moviepy --upgrade

```bash
Performing SpatioTemporal Action Detection for each clip
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  ] 27/28, 13.7 task/s, elapsed: 2s, ETA:     0sPerforming visualization
Moviepy - Building video include/mmaction2/demo/demo_det_onnx.mp4.
Moviepy - Writing video include/mmaction2/demo/demo_det_onnx.mp4

Traceback (most recent call last):
  File "./include/mmaction2/demo/demo_spatiotemporal_det_onnx.py", line 358, in <module>
    main()
  File "./include/mmaction2/demo/demo_spatiotemporal_det_onnx.py", line 352, in main
    vid.write_videofile(args.out_filename)
  File "/home/yuga/.local/lib/python3.8/site-packages/decorator.py", line 232, in fun
    return caller(func, *(extras + args), **kw)
  File "/usr/local/lib/python3.8/dist-packages/moviepy/decorators.py", line 54, in requires_duration
    return f(clip, *a, **k)
  File "/home/yuga/.local/lib/python3.8/site-packages/decorator.py", line 232, in fun
    return caller(func, *(extras + args), **kw)
  File "/usr/local/lib/python3.8/dist-packages/moviepy/decorators.py", line 135, in use_clip_fps_by_default
    return f(clip, *new_a, **new_kw)
  File "/home/yuga/.local/lib/python3.8/site-packages/decorator.py", line 232, in fun
    return caller(func, *(extras + args), **kw)
  File "/usr/local/lib/python3.8/dist-packages/moviepy/decorators.py", line 22, in convert_masks_to_RGB
    return f(clip, *a, **k)
  File "/usr/local/lib/python3.8/dist-packages/moviepy/video/VideoClip.py", line 300, in write_videofile
    ffmpeg_write_video(self, filename, fps, codec,
  File "/usr/local/lib/python3.8/dist-packages/moviepy/video/io/ffmpeg_writer.py", line 213, in ffmpeg_write_video
    with FFMPEG_VideoWriter(filename, clip.size, fps, codec = codec,
  File "/usr/local/lib/python3.8/dist-packages/moviepy/video/io/ffmpeg_writer.py", line 88, in __init__
    '-r', '%.02f' % fps,
TypeError: must be real number, not NoneType
```