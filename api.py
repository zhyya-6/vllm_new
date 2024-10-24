from openai import OpenAI

import os
import base64

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
client = OpenAI(
    base_url="http://localhost:8000/v1",
    # api_key=""
    api_key="token-abc123expo",
)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')



fig_path="/home/zhy/VLModel/assets"
 
for filename in os.listdir(fig_path):
    if filename.endswith('.png'):
       image_path=os.path.join(fig_path, filename)
       print(image_path)
       base64_image = encode_image(image_path)
       messages=[
        {
            "role": "user", 
             "content": [
                {"type":"text", "text":"What's in this image?"},
                {
                   "type":"image_url",
                   "image_url":{
                      "url":f"data:image/png;base64,{base64_image}"
                      }
                }
            ]
        }
        ]
       completion = client.chat.completions.create(
          model="/home/zhy/vllm/model_weights",
          messages=messages
        )
       chat_response = completion
       answer = chat_response.choices[0].message.content
       print(f'HawkLlama: {answer}')
