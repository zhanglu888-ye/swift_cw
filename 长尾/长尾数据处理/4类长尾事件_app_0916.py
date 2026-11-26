#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# === 事件配置 ===
EVENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "垃圾事件": {"categories": {"垃圾桶满溢", "垃圾堆积"}},
    "倒地事件": {"categories": {"人员异常倒地"}},
    "占道经营事件": {"categories": {"流动摊贩", "出店经营"}},
    "人群聚集事件": {"categories": {"异质文化异常聚集", "开放空间异常聚集"}},
}

# 4个顶层输出文件夹名（固定）
TOPS = ["01_original", "02_visualized", "03_vlm", "04_vqa"]

# 每类可视化颜色
EVENT_COLORS = {
    "垃圾事件": (66, 135, 245),
    "倒地事件": (245, 66, 95),
    "占道经营事件": (51, 170, 51),
    "人群聚集事件": (236, 184, 15),
    "default": (0, 255, 0), # 为未知事件类型提供默认颜色
}


def natural_key(s: str):
    """自然排序：crop_2 在 crop_10 之前"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_font(simhei_path: Path, size: int) -> ImageFont.FreeTypeFont:
    if not simhei_path.exists():
        raise FileNotFoundError(f"SimHei 字体不存在: {simhei_path}")
    return ImageFont.truetype(str(simhei_path), size=size)


def is_xyxy(b: List[float], W: int, H: int) -> bool:
    if len(b) != 4:
        return True
    x1, y1, x2, y2 = b
    if x2 > x1 and y2 > y1 and 0 <= x1 < W and 0 <= y1 < H:
        return True
    return False


def to_xyxy(b: List[float], W: int, H: int) -> Tuple[int, int, int, int]:
    if len(b) != 4:
        return 0, 0, 0, 0
    x1, y1, x2, y2 = b
    if is_xyxy(b, W, H):
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        return max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
    else:
        x, y, w, h = b
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        return max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)


def draw_bbox_and_label(
    img: Image.Image,
    boxes: List[Tuple[Tuple[int, int, int, int], str, str, int]],
    color: Tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
):
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), cat, crop_id, idx in boxes:
        for k in range(3):
            draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color, width=1)
        label = f"{idx}｜{cat}｜{crop_id}"
        # 使用 textbbox 获取精确的文本框尺寸
        try:
            _, _, tw, th = draw.textbbox((0, 0), label, font=font)
        except AttributeError: # 兼容旧版Pillow
             tw, th = draw.textsize(label, font=font)
        pad = 3
        bx1, by1 = x1, max(0, y1 - th - 2 * pad)
        bx2, by2 = x1 + tw + 2 * pad, by1 + th + 2 * pad
        draw.rectangle([bx1, by1, bx2, by2], fill=(0, 0, 0))
        draw.text((bx1 + pad, by1 + pad), label, fill=(255, 255, 255), font=font)


def match_event_type(category_name: str) -> Optional[str]:
    for et, cfg in EVENT_CONFIG.items():
        if category_name in cfg["categories"]:
            return et
    return None

# -----------------------
# **VLM 聚合：核心修改**
# -----------------------
def find_and_merge_vlm_jsons(vlm_root: Path, case: str) -> Dict[str, Any]:
    """
    查找给定 case 的所有 VLM JSON 文件，并将它们的 'event_analyses' 列表合并。
    """
    candidate_paths = list(vlm_root.glob(f"**/{case}*.json"))
    unique_paths = sorted(list(set(p.resolve() for p in candidate_paths if p.is_file())))

    if not unique_paths:
        return {"case_id": case, "status": "not_found", "message": "No VLM JSON files found for this case."}

    all_event_analyses = []
    found_sources = []
    
    for path in unique_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "event_analyses" in data and isinstance(data["event_analyses"], list):
                all_event_analyses.extend(data["event_analyses"])
                found_sources.append(str(path.relative_to(vlm_root)))
            else:
                logging.warning(f"[{case}] VLM JSON {path.name} 格式不正确，已跳过。")
        except Exception as e:
            logging.warning(f"[{case}] VLM JSON 解析失败 {path.name}: {e}")

    if not all_event_analyses:
        return {
            "case_id": case,
            "status": "no_valid_data",
            "message": "找到JSON文件，但没有包含有效的 'event_analyses' 数据。",
            "sources_checked": [str(p.relative_to(vlm_root)) for p in unique_paths]
        }

    # 可选：按 crop_id 对合并后的结果排序，确保一致性
    all_event_analyses.sort(key=lambda x: natural_key(x.get("crop_id", "")))

    merged_payload = {
        "case_id": case,
        "event_analyses": all_event_analyses,
        "count": len(all_event_analyses),
        "sources": sorted(list(set(found_sources))) # 记录来源文件
    }
    return merged_payload


# -----------------------
# **VQA 聚合**
# -----------------------
CROP_RE = re.compile(r"(crop_\d+)", re.IGNORECASE)

def list_vqa_jsons(vroot: Path, case: str, event_type: str) -> List[Path]:
    found: List[Path] = []
    patterns = [
        vroot / event_type / f"{case}_*crop_*.json",
        vroot.glob(f"**/{case}_*{event_type}*crop_*.json"),
        vroot.glob(f"**/{case}*crop_*.json"),
    ]
    for patt in patterns:
        if isinstance(patt, Path):
            if patt.exists() and patt.is_file():
                 found.append(patt)
        else:
            for p in patt:
                if p.suffix.lower() == ".json" and p.is_file():
                    found.append(p)

    uniq = []
    seen = set()
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    return uniq


def extract_crop_id(path: Path, payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for k in ["crop_id", "cropId", "cropID"]:
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    m = CROP_RE.search(path.stem)
    if m:
        return m.group(1)
    return None


def write_vqa_merged_json(vroot: Path, case: str, event_type: str, out_path: Path):
    paths = list_vqa_jsons(vroot, case, event_type)
    ensure_dir(out_path.parent)

    if not paths:
        placeholder = {"status": "not_found", "message": "no vqa crop jsons", "case_id": case, "event_type": event_type}
        out_path.write_text(json.dumps(placeholder, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    items = []
    for p in paths:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning(f"[{case}/{event_type}] VQA 解析失败：{p} - {e}")
            continue
        cid = extract_crop_id(p, payload)
        if not cid:
            logging.warning(f"[{case}/{event_type}] 无法识别 crop_id：{p.name}，已跳过")
            continue
        items.append({"crop_id": cid, "data": payload})

    if not items:
        placeholder = {"status": "not_found", "message": "no valid vqa crop jsons", "case_id": case, "event_type": event_type}
        out_path.write_text(json.dumps(placeholder, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    items.sort(key=lambda it: natural_key(it["crop_id"]))
    merged = {"case_id": case, "event_type": event_type, "items": items, "count": len(items)}
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


# =======================
#           主流程
# =======================
def main():
    ap = argparse.ArgumentParser(description="批量导出：原图/可视化/VLM/VQA 四套结构（VLM/VQA 聚合）")
    ap.add_argument("--large_image_dir",
                    default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare/large_images")
    ap.add_argument("--cropped_data_dir",
                    default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare/cropped_data")
    ap.add_argument("--vlm_json_dir",
                    default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_0913")
    ap.add_argument("--vqa_json_dir",
                    default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_vqa")
    ap.add_argument("--export_root", default="/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_appdata_0916", help="输出根目录（将生成 01/02/03/04 四个顶层）")
    ap.add_argument("--simhei_path", default="/host/ssddata/disk2/zhanglu/SimHei.ttf", help="SimHei.ttf 的完整路径")
    args = ap.parse_args()

    large_image_dir = Path(args.large_image_dir)
    cropped_data_dir = Path(args.cropped_data_dir)
    vlm_root = Path(args.vlm_json_dir)
    vqa_root = Path(args.vqa_json_dir)
    export_root = Path(args.export_root)
    simhei_path = Path(args.simhei_path)

    for top in TOPS:
        for event_type in EVENT_CONFIG.keys():
            ensure_dir(export_root / top / event_type)

    try:
        font = load_font(simhei_path, size=18)
    except Exception as e:
        logging.error(f"加载 SimHei 字体失败：{e}")
        return

    case_dirs = [d for d in cropped_data_dir.iterdir() if d.is_dir()]
    logging.info(f"发现 {len(case_dirs)} 个案例，开始处理...")

    for case_dir in sorted(case_dirs, key=lambda p: p.name):
        case = case_dir.name

        large_img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            t = next(large_image_dir.glob(f"{case}{ext}"), None)
            if t:
                large_img_path = t
                break
        if not large_img_path:
            logging.warning(f"[{case}] 找不到大图，跳过该案例")
            continue

        events_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for sub in sorted([p for p in case_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            for meta in sorted(sub.glob("*.json"), key=lambda p: natural_key(p.stem)):
                img = meta.with_suffix(".jpg")
                if not img.exists():
                    alt_png = meta.with_suffix(".png")
                    img = alt_png if alt_png.exists() else None
                if not img:
                    continue
                try:
                    meta_data = json.loads(meta.read_text(encoding="utf-8"))
                except Exception:
                    logging.warning(f"[{case}] 元数据无法读取: {meta}")
                    continue

                cat = meta_data.get("category_name")
                bbox = meta_data.get("original_bbox")
                if not cat or not bbox:
                    continue
                event_type = match_event_type(cat)
                if not event_type:
                    continue
                crop_id = Path(img).stem
                events_by_type[event_type].append(
                    {"crop_id": crop_id, "category_name": cat, "bbox": bbox}
                )

        if not events_by_type:
            logging.info(f"[{case}] 没有落在 4 类事件内，跳过。")
            continue

        # --- 聚合资源生成 (为当前 case 生成一次) ---

        # 1. 生成包含所有事件框的可视化图
        visualized_image = None
        try:
            with Image.open(large_img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
                for event_type, items in events_by_type.items():
                    items_sorted = sorted(items, key=lambda x: natural_key(x["crop_id"]))
                    boxes = []
                    for idx, it in enumerate(items_sorted, start=1):
                        xyxy = to_xyxy(it["bbox"], W, H)
                        boxes.append((xyxy, it["category_name"], it["crop_id"], idx))
                    
                    color = EVENT_COLORS.get(event_type, EVENT_COLORS["default"])
                    draw_bbox_and_label(im, boxes, color, font)
                visualized_image = im
        except Exception as e:
            logging.warning(f"[{case}] 生成聚合可视化图失败: {e}")
            visualized_image = None

        # 2. 【新逻辑】合并所有与 case 相关的 VLM JSON 的内容
        merged_vlm_payload = find_and_merge_vlm_jsons(vlm_root, case)
        
        # --- 资源分发 (为每个关联的事件类型输出文件) ---
        
        for event_type in events_by_type.keys():
            # 1) 原图
            dst_img1 = export_root / "01_original" / event_type / f"{case}{large_img_path.suffix}"
            ensure_dir(dst_img1.parent)
            if not dst_img1.exists():
                try:
                     shutil.copy2(large_img_path, dst_img1)
                except Exception as e:
                    logging.warning(f"[{case}/{event_type}] 拷贝原图失败: {e}")

            # 2) 可视化 (保存已生成的聚合图)
            if visualized_image:
                dst_img2 = export_root / "02_visualized" / event_type / f"{case}.jpg"
                try:
                    ensure_dir(dst_img2.parent)
                    visualized_image.save(dst_img2, quality=95)
                except Exception as e:
                     logging.warning(f"[{case}/{event_type}] 保存可视化图失败: {e}")

            # 3) VLM (写入已合并的 JSON)
            vlm_dst = export_root / "03_vlm" / event_type / f"{case}.json"
            ensure_dir(vlm_dst.parent)
            try:
                vlm_dst.write_text(json.dumps(merged_vlm_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                logging.warning(f"[{case}/{event_type}] 写入合并后 VLM JSON 失败: {e}")

            # 4) VQA (聚合 per-crop)
            vqa_dst = export_root / "04_vqa" / event_type / f"{case}.json"
            write_vqa_merged_json(vqa_root, case, event_type, vqa_dst)

        logging.info(f"[{case}] OK, 包含事件: {list(events_by_type.keys())}")

    logging.info("全部完成！")


if __name__ == "__main__":
    main()