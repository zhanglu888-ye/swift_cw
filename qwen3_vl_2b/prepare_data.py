#!/usr/bin/env python3
"""Convert supplied ShareGPT splits into ms-swift SFT and GRPO JSONL."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from schema import LABELS, parse_answer


def prepare(source, image_root, output, check_images=True):
    prepared, report, seen = {}, {}, {}
    for split in ('train', 'val'):
        rows = json.loads((source / f'{split}.json').read_text())
        sft, rl, counts = [], [], collections.Counter()
        for index, row in enumerate(rows):
            messages = row['messages']
            if len(messages) != 2 or [m['role'] for m in messages] != ['user', 'assistant']:
                raise ValueError(f'{split}:{index}: 需要一轮 user/assistant 对话')
            answer = parse_answer(messages[-1]['content'])
            if len(row['images']) != 1 or messages[0]['content'].count('<image>') != 1:
                raise ValueError(f'{split}:{index}: 需要单张图片和一个 <image>')
            old = Path(row['images'][0])
            # Preserve positive/negative, never resolve by basename alone.
            if old.parent.name not in ('positive', 'negative'):
                raise ValueError(f'未知图片目录: {old}')
            path = image_root / old.parent.name / old.name
            key = str(path)
            if check_images:
                if not path.is_file():
                    raise FileNotFoundError(f'请同步原始图片到服务器: {path}')
                key = hashlib.sha256(path.read_bytes()).hexdigest()
            if key in seen:
                raise ValueError(f'重复图片或跨划分泄漏: {path}, 首次出现 {seen[key]}')
            seen[key] = f'{split}:{index}'
            text = json.dumps(answer, ensure_ascii=False)
            sft.append({'messages': [messages[0], {'role': 'assistant', 'content': text}],
                        'images': [str(path)]})
            rl.append({'messages': [messages[0]], 'images': [str(path)], 'solution': text})
            counts.update(f'{k}:{answer[k]}' for k in LABELS)
        prepared[split] = (sft, rl)
        report[split] = {'samples': len(rows), 'labels': dict(counts)}
    output.mkdir(parents=True, exist_ok=True)
    for split, datasets in prepared.items():
        for kind, rows in zip(('sft', 'grpo'), datasets):
            with (output / f'{kind}_{split}.jsonl').open('w') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
    report['images_checked'] = check_images
    (output / 'stats.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--image-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--skip-image-check', action='store_true', help='仅检查标注结构，不能代替服务器校验')
    a = p.parse_args()
    prepare(a.source.resolve(), a.image_root.resolve(), a.output.resolve(), not a.skip_image_check)
