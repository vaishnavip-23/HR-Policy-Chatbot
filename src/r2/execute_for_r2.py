from r2_client import upload_pdf, parse_pdf, upload_parsed_files
from pathlib import Path

# Define your documents and their IDs
DOCUMENTS = {
    "CHEMEXCIL_hr_policy.pdf": "chemexcil",
    "IIMA_hr_policy.pdf": "iima", 
    "tccap_hr_policy.pdf": "tccap"
}

DOC_DIR = Path("doc")

# Process each document
for doc_name, doc_id in DOCUMENTS.items():
    doc_path = DOC_DIR / doc_name
    
    print(f"\n--- Processing {doc_id} ---")
    
    # 1) Upload raw PDF
    print(f"Uploading PDF for {doc_id}...")
    pdf_key = upload_pdf(str(doc_path), doc_id)
    print(f"✓ Uploaded to: {pdf_key}")
    
    # 2) Parse from R2 → markdown + page_map in memory
    print(f"Parsing PDF {doc_id}...")
    markdown_text, page_map = parse_pdf(doc_id)
    print(f"✓ Parsed {len(page_map)} pages")
    
    # 3) Store parsed artifacts back in R2
    print(f"Uploading parsed files for {doc_id}...")
    markdown_key, page_map_key = upload_parsed_files(doc_id, markdown_text, page_map)
    print(f"✓ Markdown uploaded to: {markdown_key}")
    print(f"✓ Page map uploaded to: {page_map_key}")
    
    print(f"✓ Complete! All files processed for {doc_id}")

print("\n✓ All documents processed!")