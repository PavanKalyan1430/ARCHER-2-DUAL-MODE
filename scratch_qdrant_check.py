from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collections = client.get_collections()
print("Collections:", collections)

points = client.scroll(
    collection_name="archer_documents",
    limit=100,
    with_payload=True,
    with_vectors=False
)

print("\n--- Scroll Points (Payloads) ---")
for p in points[0]:
    if p.payload.get("filename") == "mock_solar_system.docx":
        print(f"Point ID: {p.id}")
        print(f"Payload: {p.payload}")
