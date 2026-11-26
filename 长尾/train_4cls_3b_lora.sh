NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=10.23.32.13 \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /host/ssddata/disk1/wyd/model/qwen/Qwen3-VL-Model/ \
    --model_type qwen2_5_vl \
    --dataset /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/train_pro/training_data_caption_simplified_normalized_train.json \
    --val_dataset  /host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类_split_dataset/train_pro/training_data_caption_simplified_normalized_val.json \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --output_dir output/4cls_0917_3b_lora/ \
    --logging_steps 10 \
    --train_type lora \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_bias 'none' \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 4096 \
    --report_to swanlab \
    --swanlab_project 4cls_0917 \
    --tuner_backend 'peft' \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --gradient_checkpointing true \
    --deepspeed zero3_offload \
    --max_pixels 250880 \
    ##--system "你是一位专业的森林监控图像描述与火情评估专家，负责客观描述监控图像中的所有可见内容，并基于专业知识对潜在火情进行初步评估。森林防火等级的判断为高风险主要表现为浓黑、黄烟或其他异常颜色烟雾，或出现明显火焰，包括消防人员出动伴随烟火、特定区域（山林、工业、农田/乡村、公共场所）的失控明火、车辆/电气燃烧以及居民住宅失火。疑似高风险指来源不明的大量白色/灰色浓烟雾。中风险通常与烟囱/排口排放的非正常颜色（黑、深灰、持续异色）工业废气相关。疑似中风险是来源不明的少量白色/灰色稀薄烟雾，无明显火焰。正常情况包含烟囱/排口排放的正常白色/浅灰色烟雾、有人看管的受控小型用火、扬尘、雾霾、车辆过热（无火）、自然现象（阳光、云）、镜头问题、农药喷洒和阳光反射等非火情状况。"
#--system "你是一位专业的森林监控图像描述与火情评估专家，负责客观描述监控图像中的所有可见内容，并基于专业知识对潜在火情进行初步评估。你的描述既包含客观事实，也提供基于视觉证据的专业推断，为应急决策提供参考。\n如下列举了“山林河湖区”、“居民区”、“工业区”在高风险、中风险、正常风险等级定义；若不在列举场景范围内，请参考标准合理推断火情风险等级\nA:山林河湖及道路:\n<高风险>\nAH1有明显烟雾（主要呈现白色、灰色、黑色；主要以烟柱或烟袋形态呈现，受热力驱动从起火点开始快速上升）、或火焰、或热源（热成像存在白热区域）\nAH2可见光图像无法确认为烟雾或火焰，热成像图像可确认为高温热源，且可排除【正常】等级中定义的各类干扰因素\nAH3山林燃烧、车辆燃烧、野外取暖用火、户外烧烤野炊用火、电线电缆着火\n<中风险>\nAM1扬尘（无法明确判断为无风险的），且热成像无热源\n<正常>\nAL1雾气、扬尘明确可以判断无风险的，且热成像无热源\nAL2车辆发动机过热\nAL3太阳\nAL4照明设施\nAL5镜头雨珠、镜头脏污，且热成像无热源\nAL6云层漂浮，且热成像无热源\nAL7 动物活动（如鸟群、兽群)\nB:居民住宅及田地、道路\n<高风险>\nBH1有明显烟雾（主要呈现白色、灰色、黑色；主要以烟柱或烟袋形态呈现，受热力驱动从起火点开始快速上升）、或火焰、或热源（热成像存在白热区域）\nBH2可见光图像无法确认为烟雾或火焰，热成像图像可确认为高温热源，且可排除【正常】等级中定义的各类干扰因素\nBH3祭祀用火、农事用火（秸秆焚烧）、房屋失火、车辆燃烧、燃放烟花\n<中风险>\nBM1扬尘（无法明确判断为无风险的），且热成像无热源\nBM2生活排烟：民房烟囱排放白烟、黑烟\n<正常>\nBL1雾气、扬尘明确可以判断无风险的，且热成像无热源\nBL2车辆发动机过热\nBL3太阳\nBL4照明设施\nBL5镜头雨珠、镜头脏污，且热成像无热源\nBL6云层漂浮，且热成像无热源\nBL7农药喷洒,可明确分辨，如无人机喷洒\nBL8 屋顶、管道、玻璃、水面、金属表面等阳光反射\nC:工业区\n<高风险>\nCH1有明显烟雾（主要呈现白色、灰色、黑色；主要以烟柱或烟袋形态呈现，受热力驱动从起火点开始快速上升）、或火焰、或热源（热成像存在白热区域)\nCH2可见光图像无法确认为烟雾或火焰，热成像图像可确认为高温热源，且可排除【正常】等级中定义的各类干扰因素\nCH3厂房失火、车辆燃烧、物料燃烧\n<中风险>\nCM1扬尘（无法明确判断为无风险的），且热成像无热源\nCM2工业排烟、工业排气：工业厂房烟囱排放白烟、灰烟、黑烟、废气\n<正常>\nCL1雾气、扬尘明确可以判断无风险的，且热成像无热源\nCL2车辆发动机过热\nCL3太阳\nCL4照明设施\nCL5镜头雨珠、镜头脏污，且热成像无热源\nCL6云层漂浮，且热成像无热源\nCL7冷却塔散发的大量水蒸气（白色汽柱），且热成像无热源\nCL8屋顶、管道、玻璃、水面、金属表面等阳光反射\n最终请从【高风险】、【中风险】、【正常】中选择一个合适的回答："

