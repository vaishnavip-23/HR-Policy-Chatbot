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

def parse_pdf(doc_id:str)->Tuple[str,Dict[str,Any]]:
    pdf_key=f"{doc_id}/source.pdf"

    presigned_url=r2_client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket":R2_BUCKET,
            "Key":pdf_key
        },
        ExpiresIn=900 # 15 minutes
    )

    docs=parser.load_data(presigned_url)
    markdown_text = "\n\n".join(d.text for d in docs)

    page_map = {}
    running_offset = 0  # cumulative character offset across pages
    for i, d in enumerate(docs):
        doc_text_len = len(d.text) if hasattr(d, "text") else 0
        start_offset = running_offset
        end_offset = start_offset + doc_text_len - 1 if doc_text_len else start_offset
        page_map[i] = {
            "page": i + 1,
            "document_index": i,
            "text_length": doc_text_len,
            "start_offset": start_offset,
            "end_offset": end_offset
        }
        running_offset += doc_text_len
    
    return markdown_text, page_map

def upload_parsed_files(doc_id:str,markdown_text:str,page_map:Dict[str,Any]):
    base_prefix=f"{doc_id}/"
    markdown_key=base_prefix + "markdown.md"
    page_map_key=base_prefix + "page_map.json"

    r2_client.put_object(
        Bucket=R2_BUCKET,
        Key=markdown_key,
        Body=markdown_text.encode("utf-8"),
        ContentType="text/markdown"
    )

    r2_client.put_object(
        Bucket=R2_BUCKET,
        Key=page_map_key,
        Body=json.dumps(page_map,indent=2).encode("utf-8"),
        ContentType="application/json"
    )
    return markdown_key, page_map_key

def download_parsed_files(doc_id: str) -> Tuple[str, Dict[str, Any]]:

    base_prefix = f"{doc_id}/"
    markdown_key = base_prefix + "markdown.md"
    page_map_key = base_prefix + "page_map.json"

    # download markdown
    md_obj = r2_client.get_object(Bucket=R2_BUCKET, Key=markdown_key)
    markdown_text = md_obj["Body"].read().decode("utf-8")

    # download page_map
    page_map_obj = r2_client.get_object(Bucket=R2_BUCKET, Key=page_map_key)
    page_map = json.loads(page_map_obj["Body"].read().decode("utf-8"))

    return markdown_text, page_map