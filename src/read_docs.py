with open("docs/lua.txt") as f:
    docs = f.read()

chunk_size = 1000

chunks = [docs[i : i + chunk_size] for i in range(0, len(docs), chunk_size)]

print("Document characters:", len(docs))
print("Number of chunks:", len(chunks))

print("\nFirst chunk:\n")
print(chunks[0])
