from rag import retrieve

question = "How do I run Lua code after reading a Markdown file?"

chunks = retrieve(question)

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk[:1000])
