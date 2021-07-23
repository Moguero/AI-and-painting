from pathlib import Path

import boto3
from botocore.exceptions import ClientError

FILE_PATH = "C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/sorted_images/kept/Water/_DSC0222.JPG"


def upload_file(file_path: Path, bucket: str, object_name=None) -> None:
    """Upload a file to a specific bucket.

    :param file_path : The pathname of the file to be uploaded.
    :param bucket : The name of the bucket where to upload the file.
    :param object_name : The name of the file in the S3 bucket.
    """
    # Use file_name if S3 object_name wasn't specified
    if object_name is None:
        object_name = file_path

    # Upload the file
    s3_client = boto3.client("s3")
    # todo : check if bucket name exists
    try:
        s3_client.meta.client.upload_file(file_path, bucket, object_name)
    except ClientError as e:
        print(e)
