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
    "prompt":  prompt_data,
    "max_tokens": 512,
    "temperature": 0.8,
    "topP": 0.8,
    }
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

body = json.dumps(payload)
response = bedrock_client.invoke_model(
    body=body,
    modelId=model_id,
    contentType="application/json",
    accept="application/json",
)

response_body = json.loads(response["body"].read())
response_text = response_body["completions"][0].get("data").get("text")
print("response_body:", response_body)
print("response:", response_text)
