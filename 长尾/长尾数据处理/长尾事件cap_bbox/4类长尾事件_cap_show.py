#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量三栏可视化（输出到 --out_dir，并在其下创建 4 个事件子文件夹）
- 左：主图 + 本类的全部 bbox（标签“序号. 类别 (crop_id)”）
- 中：该类的 VLM 聚合（<vlm_dir>/<事件类型>/<case>_<事件类型>.json）
- 右：该类的 VQA 聚合（<vqa_dir>/VQA_<事件类型>/*<case>_crop_*.json）

保存路径：<out_dir>/<事件类型>/<case>_viz.png

示例：
python viz_three_panel_batch_cn_outdir.py \
  --simhei ./SimHei.ttf \
  --large_image_dir "/host/data/.../large_images" \
  --cropped_data_dir "/host/data/.../cropped_data" \
  --vlm_dir "/host/data/.../4类长尾人工标注结果_vlm_0913" \
  --vqa_dir "/host/data/.../4类长尾人工标注结果_vlm_vqa" \
  --out_dir "/host/data/.../viz_out" \
  --limit 50
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager
from PIL import Image

# ---------- 4 类事件 & 类别映射 ----------
EVENT_TYPES = ("倒地事件", "垃圾事件", "人群聚集事件", "占道经营事件")
EVENT_CATEGORIES: Dict[str, set] = {
    "倒地事件": {"人员异常倒地"},
    "垃圾事件": {"垃圾桶满溢", "垃圾堆积"},
    "人群聚集事件": {"异质文化异常聚集", "开放空间异常聚集"},
    "占道经营事件": {"流动摊贩", "出店经营"},
}

# ---------- 类别配色 ----------
CATEGORY_COLORS = {
    "出店经营": "#E67E22",
    "流动摊贩": "#E74C3C",
    "垃圾桶满溢": "#8E44AD",
    "垃圾堆积": "#9B59B6",
    "人员异常倒地": "#2E86C1",
    "异质文化异常聚集": "#16A085",
    "开放空间异常聚集": "#1ABC9C",
}
MONO_FONT_KW = dict(fontsize=9)

# ---------- 字体 ----------
def setup_chinese_font(simhei_path: Path):
    if not simhei_path.exists():
        raise FileNotFoundError(f"未找到 SimHei.ttf：{simhei_path}")
    font_manager.fontManager.addfont(str(simhei_path))
    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False
    return font_manager.FontProperties(fname=str(simhei_path))

# ---------- 小工具 ----------
def natural_crop_key(name: str) -> Tuple[int, int]:
    m = re.search(r"crop_(\d+)", name, re.I)
    return (0, int(m.group(1))) if m else (1, 10**9)

def parse_bbox(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    """统一成 (x, y, w, h)；兼容 [x1,y1,x2,y2] / [x,y,w,h] / dict。"""
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            return float(x1), float(y1), float(x2 - x1), float(y2 - y1)
        if all(k in bbox for k in ("x", "y", "w", "h")):
            return float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, a, b = bbox
        if a > x1 and b > y1 and a > 1 and b > 1:
            return float(x1), float(y1), float(a - x1), float(b - y1)
        return float(x1), float(y1), float(a), float(b)
    return None

def load_large_image(large_image_dir: Path, case: str) -> Optional[Image.Image]:
    m = sorted(large_image_dir.glob(f"{case}.*"))
    if not m:
        return None
    return Image.open(m[0]).convert("RGB")

def collect_crops(cropped_data_dir: Path, case: str) -> List[Dict[str, Any]]:
    """收集该 case 下所有小图（按 crop_序号排序）。"""
    case_dir = cropped_data_dir / case
    if not case_dir.is_dir():
        return []
    items: List[Dict[str, Any]] = []
    for sub in sorted([d for d in case_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
        for meta in sorted(sub.glob("*.json"), key=lambda p: natural_crop_key(p.stem)):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                continue
            img_path = meta.with_suffix(".jpg")
            m = re.search(r"(crop_\d+)", img_path.stem, re.I)
            crop_id = m.group(1).lower() if m else None
            if not crop_id:
                continue
            bbox = parse_bbox(data.get("original_bbox"))
            items.append({
                "crop_id": crop_id,
                "category_name": data.get("category_name", ""),
                "bbox": bbox
            })
    items.sort(key=lambda d: natural_crop_key(d["crop_id"]))
    return items

def load_vlm_case_type(vlm_dir: Path, case: str, event_type: str) -> List[Dict[str, Any]]:
    """读取某 case + 事件类型的 VLM 结果列表（event_analyses）。"""
    f = vlm_dir / event_type / f"{case}_{event_type}.json"
    if not f.exists():
        return []
    try:
        obj = json.loads(f.read_text(encoding="utf-8"))
        lst = obj.get("event_analyses", [])
        return lst if isinstance(lst, list) else []
    except Exception:
        return []

def load_vqa_case_type(vqa_dir: Path, case: str, event_type: str) -> List[Dict[str, Any]]:
    """读取某 case + 事件类型的 VQA 结果（按 crop 序号排序）。"""
    et_dir = vqa_dir / f"VQA_{event_type}"
    if not et_dir.exists():
        return []
    files = sorted(et_dir.glob(f"*{case}*crop_*.json"), key=lambda p: natural_crop_key(p.stem))
    acc: List[Dict[str, Any]] = []
    for p in files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            payload = obj if isinstance(obj, dict) else {"data": obj}
            acc.append({"__file__": p.name, **payload})
        except Exception:
            continue
    return acc

def multiline_json_preview(obj: Any, max_chars: int = 18000) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    return s if len(s) <= max_chars else s[:max_chars] + "\n...\n(内容过长，已截断)"

# ---------- 绘图 ----------
def draw_left(ax, image: Image.Image, crops: List[Dict[str, Any]], allow_categories: set, font_prop):
    ax.imshow(image)
    ax.axis("off")
    filtered = [c for c in crops if c.get("category_name") in allow_categories]
    for idx, c in enumerate(filtered, start=1):
        bbox = c.get("bbox")
        if not bbox:
            continue
        x, y, w, h = bbox
        color = CATEGORY_COLORS.get(c.get("category_name", ""), "#2ECC71")
        ax.add_patch(Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"))
        label = f"{idx}. {c.get('category_name','')} ({c.get('crop_id','')})"
        ax.text(x, max(y - 5, 0), label, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc=color, ec=color, alpha=0.2),
                color=color, fontsize=10, weight="bold", fontproperties=font_prop)

def render_case_event(case: str,
                      event_type: str,
                      large_image_dir: Path,
                      cropped_data_dir: Path,
                      vlm_dir: Path,
                      vqa_dir: Path,
                      out_dir: Path,
                      font_prop) -> bool:
    img = load_large_image(large_image_dir, case)
    if img is None:
        print(f"[SKIP] 主图未找到：{case}")
        return False

    crops_all = collect_crops(cropped_data_dir, case)
    allow_categories = EVENT_CATEGORIES.get(event_type, set())
    vlm_list = load_vlm_case_type(vlm_dir, case, event_type)
    vqa_list = load_vqa_case_type(vqa_dir, case, event_type)

    mid_text = "未找到该类的 VLM 结果。" if not vlm_list else multiline_json_preview({"event_analyses": vlm_list})
    right_text = "未找到该类的 VQA 结果。" if not vqa_list else multiline_json_preview({
        "files": [x.get('__file__','') for x in vqa_list],
        "payload": vqa_list
    })

    fig = plt.figure(figsize=(20, 10))
    gs = plt.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.03)

    ax_left = plt.subplot(gs[0, 0])
    draw_left(ax_left, img, crops_all, allow_categories, font_prop)
    ax_left.set_title("主图 + BBoxes（序号. 类别 (crop_id)）", fontsize=12, fontproperties=font_prop, weight="bold")

    ax_mid = plt.subplot(gs[0, 1]); ax_mid.axis("off")
    ax_mid.set_title("VLM 描述 JSON（仅本类）", fontsize=12, fontproperties=font_prop, weight="bold")
    ax_mid.text(0.01, 0.98, mid_text, ha="left", va="top", fontproperties=font_prop, **MONO_FONT_KW)

    ax_right = plt.subplot(gs[0, 2]); ax_right.axis("off")
    ax_right.set_title("VQA 结果 JSON（仅本类）", fontsize=12, fontproperties=font_prop, weight="bold")
    ax_right.text(0.01, 0.98, right_text, ha="left", va="top", fontproperties=font_prop, **MONO_FONT_KW)

    plt.suptitle(f"{case}  /  {event_type}", fontsize=14, fontproperties=font_prop, weight="bold")

    # 保存到 <out_dir>/<事件类型>/<case>_viz.png
    et_out = out_dir / event_type
    et_out.mkdir(parents=True, exist_ok=True)
    out_path = et_out / f"{case}_viz.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] {event_type} -> {out_path}")
    return True

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser(description="批量三栏可视化（遍历四类、无需 case）")
    ap.add_argument("--simhei", default="/host/ssddata/disk2/zhanglu/SimHei.ttf", help="本地 SimHei.ttf 路径（必需可读）")
    ap.add_argument("--large_image_dir",
                    default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare/large_images")
    ap.add_argument("--cropped_data_dir",
                    default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare/cropped_data")
    ap.add_argument("--vlm_dir",
                    default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_0913")
    ap.add_argument("--vqa_dir",
                    default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_vqa")
    ap.add_argument("--out_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_show")
    ap.add_argument("--pattern", default="", help="仅处理包含该子串的 case 名（可选）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少个 case（0=不限制）")
    ap.add_argument("--show", action="store_true", help="同时弹窗显示（批量时不建议）")
    args = ap.parse_args()
    
    

    # 字体
    font_prop = setup_chinese_font(Path(args.simhei))

    large_image_dir = Path(args.large_image_dir)
    cropped_data_dir = Path(args.cropped_data_dir)
    vlm_dir = Path(args.vlm_dir)
    vqa_dir = Path(args.vqa_dir)
    out_dir = Path(args.out_dir)
    # 预建 4 个事件子目录
    for et in EVENT_TYPES:
        (out_dir / et).mkdir(parents=True, exist_ok=True)

    # 收集所有 case（以 cropped_data_dir 下子目录为准）
    cases = sorted([p.name for p in cropped_data_dir.iterdir() if p.is_dir()])
    if args.pattern:
        cases = [c for c in cases if args.pattern in c]
    if args.limit and args.limit > 0:
        cases = cases[:args.limit]
    if not cases:
        print("[WARN] 未找到任何 case。")
        return

    print(f"[INFO] 共 {len(cases)} 个 case；输出目录：{out_dir}/<事件类型>/")
    total_ok = 0
    for case in cases:
        for event_type in EVENT_TYPES:
            try:
                if render_case_event(case, event_type, large_image_dir, cropped_data_dir, vlm_dir, vqa_dir, out_dir, font_prop):
                    total_ok += 1
            except Exception as e:
                print(f"[ERR] {case} / {event_type}: {e}")

    print(f"[DONE] 生成图像：{total_ok} 张；根目录：{out_dir}")

if __name__ == "__main__":
    main()
    
    
    
    
def main():
    ap = argparse.ArgumentParser(description="批量三栏可视化（遍历四类、无需 case）")
    ap.add_argument("--large_image_dir",
                    default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare/large_images")
    ap.add_argument("--cropped_data_dir",
                    default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare/cropped_data")
    ap.add_argument("--vlm_dir",
                    default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_0913")
    ap.add_argument("--vqa_dir",
                    default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_vqa")
    ap.add_argument("--save_dir", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_show")
    ap.add_argument("--pattern", default="", help="仅处理包含该子串的 case 名（可选）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少个 case（0=不限制）")
    ap.add_argument("--show", action="store_true", help="同时弹窗显示（批量时不建议）")
    args = ap.parse_args()

    large_image_dir = Path(args.large_image_dir)
    cropped_data_dir = Path(args.cropped_data_dir)
    vlm_dir = Path(args.vlm_dir)
    vqa_dir = Path(args.vqa_dir)
    save_dir = Path(args.save_dir) if args.save_dir else (vlm_dir / "viz_png")

    # 收集全部 case（以 cropped_data_dir 下的子目录名为准）
    if not cropped_data_dir.exists():
        raise FileNotFoundError(f"cropped_data_dir 不存在：{cropped_data_dir}")
    cases = sorted([p.name for p in cropped_data_dir.iterdir() if p.is_dir()])
    if args.pattern:
        cases = [c for c in cases if args.pattern in c]
    if args.limit and args.limit > 0:
        cases = cases[:args.limit]

    if not cases:
        print("[WARN] 没找到任何 case。")
        return

    print(f"[INFO] 将处理 {len(cases)} 个 case，输出到：{save_dir}")
    ok, fail = 0, 0
    for case in cases:
        try:
            if render_case(case, large_image_dir, cropped_data_dir, vlm_dir, vqa_dir, save_dir, args.show):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"[ERR] {case}: {e}")
            fail += 1
    print(f"[DONE] 成功 {ok}，失败 {fail}，总计 {len(cases)}")

if __name__ == "__main__":
    main()