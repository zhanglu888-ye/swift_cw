#!/usr/bin/env python3
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--provider', choices=['modelscope', 'huggingface'], default='modelscope')
p.add_argument('--output', required=True)
p.add_argument('--revision', default='main')
a = p.parse_args()
kwargs = dict(model_id='Qwen/Qwen3-VL-2B-Instruct', local_dir=str(Path(a.output).resolve()), revision=a.revision)
if a.provider == 'modelscope':
    from modelscope import snapshot_download
else:
    from huggingface_hub import snapshot_download
    kwargs['repo_id'] = kwargs.pop('model_id')
snapshot_download(**kwargs)
