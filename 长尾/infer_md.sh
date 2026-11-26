swift infer \
  --model /host/ssddata/disk1/wyd/model/qwen/qwen2.5-vl-3b-instruct/ \
  --stream true \
  --infer_backend pt \
  --val_dataset /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/test/test_data_caption_simplified_unnormalized_0923_original_加定义.json \
  --max_new_tokens 8192 \
  --result_path test/qwen25_3b_unnorm_original/result_0923_original_model_加定义.json \
  --device_map 5
