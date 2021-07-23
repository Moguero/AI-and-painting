from pathlib import Path

import boto3


FILE_PATH = Path("C:/Users/thiba/OneDrive - CentraleSupelec/Mission_JCS_IA_peinture/images/sorted_images/kept/Water/_DSC0222.JPG")
BUCKET_NAME = "ai-and-painting"
OBJECT_NAME = "_DSC0222.JPG"


def upload_file(file_path: Path, bucket_name: str, object_name=None) -> None:
    """Upload a file to a specific bucket.

    :param file_path : The pathname of the file to be uploaded.
    :param bucket_name : The name of the bucket where to upload the file.
    :param object_name : The name of the file in the S3 bucket.
    """
    # Use file_name if S3 object_name wasn't specified
    if object_name is None:
        object_name = file_path.parts[-1]

    # Upload the file
    s3 = boto3.resource("s3")
    s3.meta.client.upload_file(file_path, bucket_name, object_name)

