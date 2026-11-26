#!/usr/bin/env python3
"""Evaluate an original or merged model with exactly the same validation prompts."""
import argparse
import json
from pathlib import Path
from schema import LABELS, VALUES, parse_answer


def metrics(rows):
    matrices = {k: {v: {p: 0 for p in (*VALUES, 'invalid')} for v in VALUES} for k in LABELS}
    valid, exact = 0, 0
    for row in rows:
        truth = parse_answer(row['solution'])
        try:
            pred = parse_answer(row['prediction'])
            valid += 1
            exact += all(pred[k] == truth[k] for k in LABELS)
        except (ValueError, TypeError):
            pred = {k: 'invalid' for k in LABELS}
        for k in LABELS:
            matrices[k][truth[k]][pred[k]] += 1
    result = {'samples': len(rows), 'json_valid_rate': valid / max(1, len(rows)),
              'six_label_exact_match': exact / max(1, len(rows)), 'per_label': {}}
    for k, matrix in matrices.items():
        states = {}
        for v in VALUES:
            tp = matrix[v][v]
            support = sum(matrix[v].values())
            predicted = sum(matrix[t][v] for t in VALUES)
            precision, recall = tp / max(1, predicted), tp / max(1, support)
            states[v] = {'precision': precision, 'recall': recall,
                         'f1': 2 * precision * recall / max(1e-12, precision + recall), 'support': support}
        result['per_label'][k] = {'states': states, 'confusion': matrix}
    result['present_macro_f1'] = sum(result['per_label'][k]['states']['有']['f1'] for k in LABELS) / len(LABELS)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='原始模型或合并后的完整权重目录')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--max-pixels', type=int, default=262144)
    a = p.parse_args()
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map='auto', attn_implementation='sdpa').eval()
    processor = AutoProcessor.from_pretrained(a.model, max_pixels=a.max_pixels)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open('w') as f:
        for line in Path(a.dataset).read_text().splitlines():
            row = json.loads(line)
            prompt = row['messages'][0]['content'].replace('<image>', '').strip()
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': row['images'][0], 'max_pixels': a.max_pixels},
                {'type': 'text', 'text': prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages, image_patch_size=16)
            inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            prediction = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            record = {'images': row['images'], 'solution': row['solution'], 'prediction': prediction}
            rows.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
    report = metrics(rows)
    output.with_suffix('.metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
