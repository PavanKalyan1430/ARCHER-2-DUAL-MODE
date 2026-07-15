import httpx
import time
import sys

# We'll create a small mock text file and query it
# Let's write a small document
doc_text = """
The Solar System is the gravitationally bound system of the Sun and the objects that orbit it.
The largest of these objects are the eight planets, which in order from the Sun are:
Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.
Jupiter is the largest planet in our Solar System, with a mass more than two and a half times that of all the other planets combined.
Earth is the third planet from the Sun and the only astronomical object known to harbor life.
"""

# Save as docx using python-docx
import docx
doc = docx.Document()
doc.add_paragraph(doc_text)
mock_path = "mock_solar_system.docx"
doc.save(mock_path)
print(f"Mock document created at: {mock_path}")

# Run upload request
print("Uploading document to backend...")
try:
    with open(mock_path, "rb") as f:
        files = {"file": (mock_path, f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        response = httpx.post("http://127.0.0.1:8000/upload", files=files, timeout=60.0)
    
    if response.status_code != 200:
        print(f"Upload failed: {response.status_code} - {response.text}")
        sys.exit(1)
        
    upload_data = response.json()
    doc_id = upload_data["doc_id"]
    print(f"Upload successful. doc_id: {doc_id}")
    
    # Poll status
    print("Waiting for document to be processed...")
    for _ in range(30):
        status_resp = httpx.get(f"http://127.0.0.1:8000/status/{doc_id}")
        status_data = status_resp.json()
        print(f"Status: {status_data['status']}, Progress: {status_data.get('progress')}%")
        if status_data["status"] == "ready":
            print(f"Document is ready. Chunk count: {status_data.get('chunk_count')}")
            break
        elif status_data["status"] == "failed":
            print(f"Ingestion failed: {status_data.get('error')}")
            sys.exit(1)
        time.sleep(2)
    else:
        print("Timeout waiting for ingestion.")
        sys.exit(1)
        
    # Run query request
    question = "Which is the largest planet in our solar system?"
    print(f"Querying: '{question}'...")
    query_payload = {
        "question": question,
        "search_mode": "quick",
        "doc_id": doc_id
    }
    query_resp = httpx.post("http://127.0.0.1:8000/query", json=query_payload, timeout=60.0)
    if query_resp.status_code == 200:
        query_data = query_resp.json()
        print("\n=== Query Result ===")
        print(f"Answer: {query_data['answer']}")
        print(f"Sources: {query_data['sources']}")
        print(f"Context chunks count: {len(query_data['context_used'])}")
        for i, c in enumerate(query_data['context_used']):
            print(f"  Chunk {i+1}: {c['text'][:150]}...")
    else:
        print(f"Query failed: {query_resp.status_code} - {query_resp.text}")
        
except Exception as e:
    print(f"Test failed with error: {e}")
