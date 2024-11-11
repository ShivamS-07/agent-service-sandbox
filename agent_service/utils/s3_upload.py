import asyncio
import hashlib
from typing import BinaryIO, List, NamedTuple

import aioboto3
import boto3

from agent_service.utils.logs import async_perf_logger


class S3FileUploadTuple(NamedTuple):
    file_object: BinaryIO
    bucket_name: str
    key_path: str


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


@async_perf_logger
async def async_upload_files_to_s3(files: List[S3FileUploadTuple]) -> None:
    """Upload files to S3 asynchronously.

    Args:
        files (List[S3FileUploadTuple]): List of tuples containing the file object,
            bucket name, and file key/path.
    """
    async with aioboto3.Session().client("s3") as s3_client:
        tasks = [
            s3_client.upload_fileobj(Fileobj=file_obj, Bucket=s3_bucket_name, Key=s3_key_path)
            for file_obj, s3_bucket_name, s3_key_path in files
        ]
        await asyncio.gather(*tasks)


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
