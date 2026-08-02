import pymupdf
import tiktoken

with pymupdf.open("the-python-bible.pdf") as doc:
    full_text = ""
    for page in doc:
        full_text += str(page.get_text())
    first_500 = full_text[:500]
    print(first_500)

encoding = tiktoken.get_encoding("cl100k_base")
token_id= encoding.encode(first_500)
print(token_id)
print(f"Number of tokens in first 500 characters: {len(token_id)}")
