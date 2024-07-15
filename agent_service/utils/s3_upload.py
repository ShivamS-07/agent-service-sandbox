import hashlib

import boto3


def upload_string_to_s3(data: str) -> str:
    hasher = hashlib.md5()
    hasher.update(data.encode("utf-8"))
    hash_hex = hasher.hexdigest()

    bucket_name = "boosted-agent-service"
    object_key = f"{hash_hex}.json"
    s3_path = f"s3://{bucket_name}/{object_key}"

    s3_client = boto3.client("s3")

    s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=data)
    return s3_path


def download_json_from_s3(s3_path: str) -> str:
    if not s3_path.startswith("s3://"):
        raise ValueError("Invalid S3 path. Must start with 's3://'")

    bucket_key = s3_path[5:].split("/", 1)
    if len(bucket_key) < 2:
        raise ValueError("Invalid S3 path. Path must include bucket and key")

    bucket, key = bucket_key[0], bucket_key[1]

    s3 = boto3.client("s3")

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except Exception as e:
        raise Exception(f"Failed to download from S3: {str(e)}")

    json_content = response["Body"].read().decode("utf-8")
    return json_content
