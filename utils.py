import json
import os

from pypdf import PdfReader

from indexing import build_metadata


# ==============================================================================
# LOAD SINGLE PDF
# ==============================================================================


def load_pdf(pdf_path):
    """Return ``(text, page_count)`` for a PDF file."""
    reader = PdfReader(pdf_path)
    pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)

    return "\n".join(pages), len(reader.pages)


# ==============================================================================
# LOAD ALL DOCUMENTS
# ==============================================================================


def load_documents(data_folder="data"):
    """Load every PDF in ``data_folder`` into a list of document dicts."""
    documents = []

    if not os.path.exists(data_folder):
        raise ValueError(f"Folder '{data_folder}' not found.")

    for file_name in sorted(os.listdir(data_folder)):
        if not file_name.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(data_folder, file_name)
        try:
            content, page_count = load_pdf(file_path)
        except Exception as exc:  # a single broken PDF should not kill the app
            print(f"Skipping '{file_name}': could not read PDF ({exc})")
            continue

        if not content.strip():
            print(f"Skipping '{file_name}': no extractable text.")
            continue

        documents.append(
            {
                "file_name": file_name,
                "source": file_name,
                "document_name": file_name,
                "content": content,
                "metadata": build_metadata(file_name, content, page_count),
            }
        )

    return documents


# Alias for backward compatibility if imported as load_document
load_document = load_documents


# ==============================================================================
# SAVE JSON
# ==============================================================================


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ==============================================================================
# LOAD JSON
# ==============================================================================


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
