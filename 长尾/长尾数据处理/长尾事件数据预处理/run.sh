# python 1-prepare_vlm_input_251126.py --json_file "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/原始标注结果/垃圾堆积/下载结果_json_316_more_20251126104740.json" --output_dir "/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/ 垃圾堆积_316" --local_image_root "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/原始标注结果/垃圾堆积/垃圾111"
python 2-数据处理_合并改标签重命名.py \
  --inputs "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/占道经营/占道经营_314" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/占道经营/占道经营_351" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/占道经营/占道经营_365" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/非机动车/非机动车_376" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/非机动车/非机动车_377" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/垃圾堆积/垃圾堆积_364" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/垃圾堆积/垃圾堆积_378" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/垃圾堆积/垃圾堆积_380" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/人员异常聚集/人员异常聚集_379" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/人员异常聚集/人员异常聚集_733" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/人员异常聚集/人员异常聚集_734" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/污水井倾倒/污水井倾倒_732" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/人员倒地/人员倒地_381" \
  "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/人员倒地/人员倒地_350" \
  --output "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/202511/vlm_prepare/合并"