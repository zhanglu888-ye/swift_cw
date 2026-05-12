swift infer \
  --model output/4cls_0923_3b_full_unnorm/v0-20250924-195802/checkpoint-945/ \
  --stream true \
  --infer_backend pt \
  --val_dataset /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/test_data_caption_simplified_unnormalized_0923.json \
  --max_new_tokens 8192 \
  --result_path test/qwen25_3b_full_unnorm/result_v0-20250924-195802_ck945.json \
  --device_map 5
