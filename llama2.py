import boto3
import json

prompt_data = """
Act as a Shakespearean. and write a poem on machine learning.
"""

bedrock_client = boto3.client(
    service_name = "bedrock-runtime",
    region_name="us-east-1",
)

payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",
    "max_gen_len": 1024,
    # "temperature": 0.5,
    "top_p": 0.9,
    }

body = json.dumps(payload)
model_id = "meta.llama3-70b-instruct-v1:0"

response = bedrock_client.invoke_model(
    body=body,
    modelId=model_id,
    contentType="application/json",
    accept="application/json",
)

response_body = json.loads(response["body"].read())
response_text = response_body["generation"]
print("response_body:", response_body)
print("response:", response_text)