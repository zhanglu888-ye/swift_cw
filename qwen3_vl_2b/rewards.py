"""ms-swift external reward plugin: strict JSON + balanced label accuracy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import LABELS, parse_answer
try:
    from swift.rewards import ORM, orms
except ImportError:
    from swift.plugin import ORM, orms


def content(completion):
    if isinstance(completion, list):
        return completion[0]['content']
    return completion


class DangerFormat(ORM):
    def __call__(self, completions, **kwargs):
        scores = []
        for completion in completions:
            try:
                parse_answer(content(completion))
                scores.append(1.0)
            except (ValueError, TypeError, KeyError, IndexError):
                scores.append(0.0)
        return scores


class DangerAccuracy(ORM):
    def __call__(self, completions, solution, **kwargs):
        if len(completions) != len(solution):
            raise ValueError('completions 和 solution 未对齐，禁止静默截断')
        scores = []
        for completion, target in zip(completions, solution):
            truth = parse_answer(target)  # Bad training labels must fail loudly.
            try:
                pred = parse_answer(content(completion))
            except (ValueError, TypeError, KeyError, IndexError):
                scores.append(-1.0)
                continue
            # Average within GT present/uncertain and absent groups, then across groups.
            # All-negative predictions cannot get 5/6 credit on a positive example.
            groups = [[k for k in LABELS if truth[k] != '无'],
                      [k for k in LABELS if truth[k] == '无']]
            matches = [sum(pred[k] == truth[k] for k in g) / len(g) for g in groups if g]
            scores.append(2 * sum(matches) / len(matches) - 1)
        return scores


orms['danger_format'] = DangerFormat
orms['danger_accuracy'] = DangerAccuracy
