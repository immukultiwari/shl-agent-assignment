"""
ingest.py — Rebuilds ChromaDB from scratch with correct metadata.
Run with: .\\venv\\Scripts\\python.exe ingest.py
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
CATALOG_PATH = "./data/shl_catalog.json"
CHROMA_PATH  = os.getenv("CHROMA_PATH", "./data/shl_chroma_db")
COLLECTION   = "shl_assessments"
BATCH_SIZE   = 50

# ──────────────────────────────────────────────
# CONSTANTS  (must be defined before functions)
# ──────────────────────────────────────────────
KEY_TO_CODE = {
    "Knowledge & Skills":           "K",
    "Personality & Behavior":       "P",
    "Ability & Aptitude":           "A",
    "Biodata & Situational Judgment": "B",
    "Simulations":                  "S",
    "Competencies":                 "C",
    "Development & 360":            "D",
    "Assessment Exercises":         "E",
}

# Confirmed pre-packaged / bundled job solutions — excluded by entity_id
# (most reliable method — name-based matching misses renamed products)
EXCLUDED_ENTITY_IDS = {
    "3930",   # Sales & Service Phone Solution
    "3931",   # Customer Service Phone Solution
    "3932",   # Sales & Service Phone Simulation
       
    "3934",   # Entry Level Cashier Solution
    "3935",   # Entry Level Customer Service (General) Solution
    "3936",   # Entry Level Hotel Front Desk Solution
    "3937",   # Entry Level Sales Solution
    "3938",   # Entry Level Technical Support Solution
   
    "4291",   # Manufac. & Indust. - Mechanical & Vigilance 8.0
    "4292",   # Manufacturing & Industrial - Mechanical Focus 8.0
    
    "4294",   # Manufacturing & Industrial - Vigilance Focus 8.0
    "4295",   # Manufacturing & Industrial - Essential Focus 8.0
}

# Belt-and-suspenders: also catch anything with these substrings
EXCLUDED_NAME_SUBSTRINGS = [
    " Solution",       # catches any future "X Solution" names
   
]

# ──────────────────────────────────────────────
# FUNCTIONS
# ──────────────────────────────────────────────
def is_in_scope(item: dict) -> bool:
    """Return False for pre-packaged job solutions and empty items."""
    # 1. Exclude by entity_id (explicit, most reliable)
    if str(item.get("entity_id", "")) in EXCLUDED_ENTITY_IDS:
        return False

    # 2. Exclude by name substring
    name = item.get("name", "")
    for substr in EXCLUDED_NAME_SUBSTRINGS:
        if substr in name:
            return False

    # 3. Skip items with no description (report shells, not real tests)
    if not item.get("description", "").strip():
        return False

    return True


def keys_to_pipe_string(keys_list: list) -> str:
    """Store keys as pipe-separated string for later retrieval."""
    return "|".join(keys_list) if keys_list else ""


def keys_to_codes(keys_list: list) -> str:
    """Convert keys list to comma-separated single-letter codes."""
    codes = [KEY_TO_CODE[k] for k in keys_list if k in KEY_TO_CODE]
    # Deduplicate while preserving order
    seen = set()
    unique = [c for c in codes if not (c in seen or seen.add(c))]
    return ",".join(unique) if unique else "K"


def build_embedding_text(item: dict) -> str:
    """
    Richer embedding text = better semantic retrieval.
    Every field here contributes to what queries this item matches.
    """
    return "\n".join([
        f"Assessment Name: {item.get('name', '')}",
        f"Test Types: {', '.join(item.get('keys', []))}",
        f"Job Levels: {item.get('job_levels_raw', '')}",
        f"Languages: {item.get('languages_raw', '')}",
        f"Duration: {item.get('duration', 'N/A')}",
        f"Description: {item.get('description', '')}",
    ])


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    # 1. Load catalog
    print("Loading catalog...")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"Total items in catalog : {len(catalog)}")

    # 2. Filter to in-scope items only
    in_scope = [item for item in catalog if is_in_scope(item)]
    print(f"In-scope items          : {len(in_scope)}")
    print(f"Excluded                : {len(catalog) - len(in_scope)}")

    # 3. Print every excluded item so you can verify nothing legitimate was dropped
    excluded = [item for item in catalog if not is_in_scope(item)]
    print("\n── Excluded items ──────────────────────────")
    for item in excluded:
        print(f"  [{item.get('entity_id')}] {item.get('name')}")
    print("────────────────────────────────────────────\n")

    # 4. Init ChromaDB — delete old collection first for clean rebuild
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        client.delete_collection(name=COLLECTION)
        print(f"Deleted existing collection '{COLLECTION}'")
    except Exception:
        print(f"No existing collection to delete")

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # 5. Batch insert
    ids, documents, metadatas = [], [], []

    for item in in_scope:
        entity_id = str(item.get("entity_id", "")).strip()
        if not entity_id:
            continue

        ids.append(entity_id)
        documents.append(build_embedding_text(item))
        metadatas.append({
            "name":            item.get("name", ""),
            "url":             item.get("link", ""),
            "keys_raw":        keys_to_pipe_string(item.get("keys", [])),
            "test_type_codes": keys_to_codes(item.get("keys", [])),
            "duration":        item.get("duration", "N/A"),
            "job_levels":      item.get("job_levels_raw", ""),
            "languages":       item.get("languages_raw", ""),
            # Truncate description — ChromaDB metadata has size limits
            "description":     item.get("description", "")[:500],
        })

        if len(ids) >= BATCH_SIZE:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            print(f"  Inserted batch of {len(ids)}...")
            ids, documents, metadatas = [], [], []

    # Insert remaining items
    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  Inserted final batch of {len(ids)}")

    # 6. Final verification
    final_count = collection.count()
    print(f"\nDone. Collection '{COLLECTION}' has {final_count} items.")

    # Spot check — query for something that should definitely be in scope
    test = collection.query(query_texts=["Java developer assessment"], n_results=1)
    print("\nSpot check result:")
    print(f"  Name      : {test['metadatas'][0][0].get('name')}")
    print(f"  test_type : {test['metadatas'][0][0].get('test_type_codes')}")
    print(f"  keys_raw  : {test['metadatas'][0][0].get('keys_raw')}")
    print(f"  url       : {test['metadatas'][0][0].get('url')}")


if __name__ == "__main__":
    main()