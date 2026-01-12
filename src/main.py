from r2.r2_client import download_parsed_files

doc_ids = ["chemexcil","iima","tccap"]
for doc_id in doc_ids:
    markdown_text, page_map = download_parsed_files(doc_id)

    



