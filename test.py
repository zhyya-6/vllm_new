from vllm import LLM
import PIL
import os
# from PIL import Image


# from vllm.multimodal.image import ImagePixelData
from vllm import SamplingParams

# 设置环境变量，将第6号GPU映射为cuda:0
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

sample_params = SamplingParams(temperature=0, max_tokens=1024)

model_path = "/home/zhy/model_weights/llava_llama3_8b_siglip_tile_finetune_hf_0518data_1.0ep"


llm = LLM(
    model=model_path,
    tokenizer=model_path,
    tokenizer_mode="slow",
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    # image_input_type="pixel_values",
    # image_token_id=128265,
    # image_input_shape="1,3,1024,1024",
    # image_feature_size=576,
    # disable_image_processor=False,
    # max_model_len=3075,
    enforce_eager=True
)
print("ok!-----------------------------------")
# prompt = f"You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n User: {'<image>'} Describe each stage of this image detail.\nAssistant:"
prompt = " <image>what is shown in this image? "

image = image = PIL.Image.open("/home/zhy/VLModel/assets/coin.png").convert("RGB")
# image = image.convert("RGB")
outputs = llm.generate(
    {"prompt": prompt, "multi_modal_data": {"image": image}}, sample_params
)
for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)