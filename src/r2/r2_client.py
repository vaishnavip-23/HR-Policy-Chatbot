import os
import json
from typing import Tuple, Dict, Any
from pathlib import Path
import boto3
from botocore.client import Config
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

R2_ENDPOINT=os.environ.get("R2_ENDPOINT")
R2_BUCKET=os.environ.get("R2_BUCKET")
R2_ACCESS_KEY=os.environ.get("R2_ACCESS_KEY")
R2_SECRET_KEY=os.environ.get("R2_SECRET_KEY")
LLAMA_API_KEY=os.environ.get("LLAMA_API_KEY")

# R2 client 

r2_client=boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="auto"
)

parser=LlamaParse(
    api_key=LLAMA_API_KEY,
    result_type="markdown",
    verbose=False
)

def upload_pdf(pdf_path:str,doc_id:str) -> str:
    """
    Upload local PDF to R2 as: {doc_id}/original.pdf
    Returns the R2 object key.
    """
    path=Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(pdf_path)
    pdf_key=f"{doc_id}/source.pdf"

    with path.open("rb") as f:
        r2_client.put_object(
            Bucket=R2_BUCKET,
            Key=pdf_key,
            Body=f,
            ContentType="application/pdf",
            Metadata={
                "doc_id": doc_id,
                "original_filename": path.name
            }
        )
    return pdf_key

