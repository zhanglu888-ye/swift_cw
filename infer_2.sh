swift infer \
  --ckpt_dir output/4cls_0918_3b_full_norm/v3-20250919-095224/checkpoint-480/ \
  --stream true \
  --infer_backend pt \
  --val_dataset /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/test_data_caption_simplified_normalized.json \
  --max_new_tokens 8192 \
  --result_path test/qwen25_3b_full_norm/result_0923.json \
  --device_map 5
