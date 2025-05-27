import boto3
import botocore.config
import json
from datetime import datetime
import os

def blog_generate_using_bedrock(blog_topic: str) -> str:
    prompt = """
    <s>[INST]Human:Write a 200 word blog post about {blog_topic}.
    Assistant:[/INST]
    </s>
    """
    
    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9,
    }
    
    try:
        bedrock = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=os.environ.get('REGION', 'us-west-1'),
                    aws_access_key_id=os.environ.get('ACCESS_KEY_ID'),
                    aws_secret_access_key=os.environ.get('SECRET_ACCESS_KEY'),
                    config= botocore.config.Config(
                            read_timeout=300,
                            retries={"max_attempts": 3, "mode": "standard"}))
        response = bedrock.invoke_model(body=json.dumps(body),
                                        modelId="meta.llama3-70b-instruct-v1:0",
                                        accept="application/json",
                                        contentType="application/json")
        response_body = response["body"].read()
        response_text = json.loads(response_body)
        blog_details = response_text["generation"]
        print("response:", response_text)
        print("blog_details:", blog_details)
        return blog_details
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
    
def save_blog_to_s3(s3_key: str, s3_bucket:str, blog_content: str):
    s3 = boto3.client("s3")
    
    try:
        s3.put_object(Bucket=s3_bucket,
                      Key=s3_key,
                      Body=blog_content.encode('utf-8'),
                      ContentType='text/plain')
    except Exception as e:
        print(f"An error occurred while saving to S3: {e}")
        return False
    
def lambda_handler(event, context):
    event = json.loads(event["body"])
    blog_topic = event.get("blog", "No blog topic provided")
    
    generated_blog = blog_generate_using_bedrock(blog_topic)
    
    try:
        if generated_blog:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            s3_key = f"blog-output/{current_time}.txt"
            s3_bucket_name = "workwithgenaiblog"
            save_blog_to_s3(s3_key, s3_bucket_name, generated_blog)
        else:
            print("No blog generated.")
    except Exception as e:
        print(f"An error occurred while processing the blog: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Failed to generate blog"})
        }
        
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Blog generated and saved successfully"})
    }