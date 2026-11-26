"""Shared annotation schema; deliberately independent of training dependencies."""
import json

LABELS = ('刀具', '斧头', '桶壶', '液化气钢瓶', '枪形物', '弩')
VALUES = ('有', '无', '不确定')


def parse_answer(text):
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {'描述', *LABELS}:
        raise ValueError('必须恰好包含描述及六个类别')
    if not isinstance(obj['描述'], str) or not obj['描述'].strip():
        raise ValueError('描述不能为空')
    if any(obj[k] not in VALUES for k in LABELS):
        raise ValueError('类别值必须为有/无/不确定')
    return obj
