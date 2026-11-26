import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Reward logic can be tested without installing GPU training dependencies.
swift = types.ModuleType('swift')
swift_rewards = types.ModuleType('swift.plugin')
swift_rewards.ORM = object
swift_rewards.orms = {}
sys.modules.setdefault('swift', swift)
sys.modules.setdefault('swift.plugin', swift_rewards)
from rewards import DangerAccuracy, DangerFormat
from schema import LABELS
from prepare_data import prepare
from evaluate import metrics


def answer(**overrides):
    data = {'描述': '可见目标。', **{k: '无' for k in LABELS}, **overrides}
    return json.dumps(data, ensure_ascii=False)


class PipelineTests(unittest.TestCase):
    def test_rewards(self):
        truth = answer(刀具='有')
        reward = DangerAccuracy()
        self.assertEqual(reward([truth], [truth]), [1.0])
        self.assertEqual(reward([answer()], [truth]), [0.0])
        self.assertEqual(reward(['not json'], [truth]), [-1.0])
        self.assertEqual(reward([answer(刀具='不确定')], [truth]), [0.0])
        self.assertEqual(reward([answer(刀具='不确定')], [answer(刀具='不确定')]), [1.0])
        with self.assertRaises(ValueError):
            reward([truth], [])
        self.assertEqual(DangerFormat()([truth, '```json\n' + truth + '\n```']), [1.0, 0.0])

    def test_conversion_and_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / 'images'
            (images / 'positive').mkdir(parents=True)
            for split, payload in [('train', b'a'), ('val', b'b')]:
                (images / 'positive' / f'{split}.jpg').write_bytes(payload)
                row = {'messages': [{'role': 'user', 'content': '<image>检查'},
                                    {'role': 'assistant', 'content': answer(刀具='有')}],
                       'images': [f'/old/mac/positive/{split}.jpg']}
                (root / f'{split}.json').write_text(json.dumps([row]))
            prepare(root, images, root / 'out')
            rl = json.loads((root / 'out/grpo_train.jsonl').read_text())
            self.assertEqual(len(rl['messages']), 1)
            self.assertNotIn('assistant', str(rl['messages']))
            self.assertEqual(rl['images'], [str(images / 'positive/train.jpg')])
            self.assertEqual(rl['solution'], answer(刀具='有'))
            (images / 'positive/val.jpg').write_bytes(b'a')
            with self.assertRaisesRegex(ValueError, '泄漏'):
                prepare(root, images, root / 'out2')
            (images / 'positive/val.jpg').unlink()
            with self.assertRaises(FileNotFoundError):
                prepare(root, images, root / 'out3')

    def test_metrics_count_invalid_as_errors(self):
        truth = answer(刀具='有')
        report = metrics([{'solution': truth, 'prediction': truth},
                          {'solution': truth, 'prediction': 'bad'}])
        self.assertEqual(report['json_valid_rate'], 0.5)
        self.assertEqual(report['six_label_exact_match'], 0.5)
        self.assertEqual(report['per_label']['刀具']['states']['有']['recall'], 0.5)


if __name__ == '__main__':
    unittest.main()
