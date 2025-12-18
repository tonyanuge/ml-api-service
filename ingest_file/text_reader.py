def read_text_file(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")
