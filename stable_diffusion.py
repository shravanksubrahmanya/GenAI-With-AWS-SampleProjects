import boto3
import json
import base64
import os

prompt_data = """
Create an image of a futuristic city skyline at sunset, with flying cars and neon lights. The city should have a mix of modern skyscrapers and green spaces, with a vibrant atmosphere. The sky should be filled with shades of orange, pink, and purple, reflecting the setting sun. The scene should be dynamic and full of energy, capturing the essence of a bustling metropolis in the future.
"""

prompt_template = [{"text": prompt_data, "weight": 1.0}]
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

payload = {
    "text_prompts": prompt_template,
    "cfg_scale": 10,
    "speed": 0,
    "steps": 50,
    "width": 1024,
    "height": 1024,
}

model_id = "stability.stable-diffusion-xl-v1:0"

response = bedrock_client.invoke_model(
    body=json.dumps(payload),
    modelId=model_id,
    contentType="application/json",
    accept="application/json",
)
response_body = json.loads(response["body"].read())
image_base64 = response_body["artifacts"][0]["base64"]
image_data = base64.b64decode(image_base64)
output_image_path = "output_image.png"
with open(output_image_path, "wb") as image_file:
    image_file.write(image_data)
print(f"Image saved to {output_image_path}")