swift infer \
  --ckpt_dir output/4cls_0923_3b_lora_unnorm/v1-20250925-022751/checkpoint-945/ \
  --stream false \
  --infer_backend pt \
  --val_dataset /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/test_data_caption_simplified_unnormalized_0923.json \
  --max_new_tokens 8192 \
  --result_path test/qwen25_3b_lora_unnorm/result_v1-20250925-022751_ck945.json \
  --device_map 5
