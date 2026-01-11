from r2_client import upload_pdf
from pathlib import Path

# Define your documents and their IDs
# documents={doc_name:doc_id}
DOCUMENTS = {
    "CHEMEXCIL_hr_policy.pdf": "chemexcil",
    "IIMA_hr_policy.pdf": "iima", 
    "tccap_hr_policy.pdf": "tccap"
}

# Path to the doc directory
DOC_DIR = Path("doc")

# Upload each PDF
for doc_name, doc_id in DOCUMENTS.items():
    doc_path = DOC_DIR / doc_name
    print(f"Uploading PDF {doc_name}...")
    pdf_key = upload_pdf(str(doc_path), doc_id)
    print(f"✓ Uploaded to: {pdf_key}")
    

print("All uploads completed!")