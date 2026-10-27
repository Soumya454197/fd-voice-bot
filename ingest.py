from pathlib import Path
import chromadb

KNOWLEDGE_FILE = Path("data/fd_knowledge.md")
DB_PATH = "chroma_db"
COLLECTION_NAME = "fd_knowledge"


def load_chunks(path):
    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n")        # makes Windows line endings safe
    parts = text.split("\n## ")
    chunks = []
    # parts[0] is the header text before the first section, so we skip it
    for part in parts[1:]:
        lines = part.strip().split("\n")
        title = lines[0].strip()
        chunks.append({"title": title, "text": part.strip()})
    return chunks


def main():
    chunks = load_chunks(KNOWLEDGE_FILE)
    print(f"Found {len(chunks)} sections")

    client = chromadb.PersistentClient(path=DB_PATH)

    # delete the old collection so running again does not create duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{"title": c["title"]} for c in chunks],
    )
    print("Done. Knowledge stored in the database.")


if __name__ == "__main__":
    main()
    