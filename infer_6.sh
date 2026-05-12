swift infer \
  --model output/4cls_0923_7b_full_unnorm/v0-20250923-180828/checkpoint-945/ \
  --stream false \
  --infer_backend pt \
  --val_dataset /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/test_data_caption_simplified_unnormalized_0923.json \
  --max_new_tokens 8192 \
  --result_path test/qwen25_7b_full_unnorm/result_0923_new.json \
  --device_map auto
