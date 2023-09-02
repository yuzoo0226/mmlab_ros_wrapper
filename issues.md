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