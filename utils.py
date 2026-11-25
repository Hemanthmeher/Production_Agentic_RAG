import os, json
from pypdf import PdfReader
from indexing import generate_document_metadata


# ==============================================================================
# LOAD SINGLE PDF
# ==============================================================================

def load_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==============================================================================
# LOAD ALL DOCUMENTS
# ==============================================================================

def load_documents(data_folder="data"):

    documents = []

    # Check if folder exists
    if not os.path.exists(data_folder):

        raise ValueError(
            f"Folder '{data_folder}' not found."
        )

    for file_name in os.listdir(data_folder):

        if file_name.endswith(".pdf"):

            file_path = os.path.join(
                data_folder,
                file_name
            )

            content = load_pdf(file_path)

            metadata = generate_document_metadata({
                "content": content
            })

            documents.append({
                "file_name": file_name,
                "content": content,
                "metadata": metadata
            })

    return documents

# ==============================================================================
# SAVE JSON
# ==============================================================================

def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


# ==============================================================================
# LOAD JSON
# ==============================================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)