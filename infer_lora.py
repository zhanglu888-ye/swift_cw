import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift


from swift.llm import safe_snapshot_download, BaseArguments

lora_adapters = safe_snapshot_download('output/4cls_0917_3b_lora/v3-20250917-183657/checkpoint-320')
args = BaseArguments.from_pretrained(lora_adapters)
print(f'args.model: {args.model}')
print(f'args.model_type: {args.model_type}')
print(f'args.template_type: {args.template}')
print(f'args.default_system: {args.system}')


# Please adjust the following lines
model = '/host/ssddata/disk1/wyd/model/qwen/Qwen3-VL-Model'
lora_checkpoint = safe_snapshot_download('output/4cls_0917_3b_lora/v3-20250917-183657/checkpoint-320')  # Change to your checkpoint_dir
template_type = None  # None: use the default template_type of the corresponding model
default_system = None  # None: use the default system prompt of the corresponding model

# Load model and dialogue template
model, tokenizer = get_model_tokenizer(model, model_type="qwen2_5_vl")
model = Swift.from_pretrained(model, lora_checkpoint)
template_type = template_type or model.model_meta.template
template = get_template(template_type, tokenizer, default_system=default_system)
engine = PtEngine.from_model_template(model, template, max_batch_size=2)
request_config = RequestConfig(max_tokens=512, temperature=0)

# Using 2 infer_requests to demonstrate batch inference
infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': '分析<image>中与“占道经营事件”相关的情况。\n请严格按照以下定义，识别图中的子事件，并评估其风险等级。\n\n子事件定义: \"流动摊贩\", \"出店经营\", \"无异常事件\"。\n风险等级定义: \"3级-高风险\", \"2级-中风险\", \"1级-低风险\", \"0级-无风险\"。\n\n请将你的分析结果以JSON对象格式输出。'}],
                 images=['/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/4类长尾人工标注结果_vlm_prepare/large_images/训练集_2134.jpg']),
]
resp_list = engine.infer(infer_requests, request_config)
query0 = infer_requests[0].messages[0]['content']
print(f'response0: {resp_list[0].choices[0].message.content}')
print(f'response1: {resp_list[1].choices[0].message.content}')
