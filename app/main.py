from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import QueryRequest, QueryResponse, UploadResponse, DocumentStatus
import os
import shutil
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("A.R.C.H.E.R")

app = FastAPI(title="A.R.C.H.E.R - Document Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_uploads"

# ==========================================
# In-memory document status registry
# Maps doc_id -> DocumentStatus
# ==========================================
_doc_registry: dict[str, DocumentStatus] = {}

# ==========================================
# Lazy-loaded singletons (initialized on first use)
# ==========================================
_services = {}

def _get_services():
    """Initialize all services lazily on first request, not at server boot."""
    if not _services:
        logger.info("🔌 Initializing services for the first time...")

        from app.services.embedding import EmbeddingService
        from app.services.llm import LLMService
        from app.db.qdrant import QdrantManager
        from app.services.retrieval import RetrievalService
        from app.agents.nodes import AgentNodes
        from app.agents.workflow import ArcherWorkflow

        logger.info("  → Loading Embedding Model (BGE)...")
        embedding_svc = EmbeddingService()

        logger.info("  → Connecting to OpenRouter + Groq LLM pool...")
        llm_svc = LLMService()

        logger.info("  → Connecting to Qdrant Vector DB...")
        qdrant_mgr = QdrantManager()

        logger.info("  → Building Retrieval Service...")
        retrieval_svc = RetrievalService(qdrant_mgr)

        logger.info("  → Assembling LangGraph Agent Brain...")
        agent_nodes = AgentNodes(llm_svc, retrieval_svc)
        workflow = ArcherWorkflow(agent_nodes)

        _services["embedding"] = embedding_svc
        _services["llm"] = llm_svc
        _services["qdrant"] = qdrant_mgr
        _services["retrieval"] = retrieval_svc
        _services["workflow"] = workflow

        logger.info("✅ All services initialized successfully!")

    return _services


@app.on_event("startup")
async def startup():
    os.makedirs(TEMP_DIR, exist_ok=True)
    logger.info("🚀 A.R.C.H.E.R. Server is READY! Visit http://127.0.0.1:8000/docs")


# ==========================================
# BACKGROUND WORKER
# ==========================================
def process_document_background(file_path: str, filename: str, doc_id: str):
    """Background worker: embeds PDF silently. Updates registry when done."""
    try:
        logger.info(f"⚙️ Background processing started: {filename}")
        if doc_id in _doc_registry:
            _doc_registry[doc_id].status = "processing"
            _doc_registry[doc_id].progress = 5.0

        services = _get_services()

        # 1. Parse and chunk
        from app.services.ingestion import IngestionService
        ingestion_svc = IngestionService()
        chunks = ingestion_svc.process_document(file_path, filename, doc_id)

        if doc_id in _doc_registry:
            _doc_registry[doc_id].progress = 35.0

        # 2. Embed and insert into Qdrant in batches
        from llama_index.core.schema import TextNode
        from llama_index.core import VectorStoreIndex

        nodes = []
        import uuid
        for c in chunks:
            meta = c.metadata.model_dump()
            meta["chunk_id"] = c.chunk_id
            
            # Qdrant point IDs must be standard UUIDs or integers.
            # We generate a deterministic UUID using uuid5 with doc_id as namespace.
            try:
                namespace_uuid = uuid.UUID(doc_id)
                point_id = str(uuid.uuid5(namespace_uuid, c.chunk_id))
            except Exception:
                point_id = str(uuid.uuid4())
                
            nodes.append(TextNode(id_=point_id, text=c.text, metadata=meta))
            
        index = VectorStoreIndex.from_vector_store(vector_store=services["qdrant"].get_store())
        
        batch_size = 25
        total_nodes = len(nodes)
        if total_nodes == 0:
            if doc_id in _doc_registry:
                _doc_registry[doc_id].progress = 95.0
        else:
            for i in range(0, total_nodes, batch_size):
                # Verify document was not deleted in mid-flight
                if doc_id not in _doc_registry:
                    logger.info(f"⚠️ Document {doc_id} was deleted during background execution. Terminating task.")
                    return
                
                batch = nodes[i:i+batch_size]
                index.insert_nodes(batch)
                
                pct = 35.0 + ((i + len(batch)) / total_nodes) * 60.0
                _doc_registry[doc_id].progress = round(min(95.0, pct), 1)

        # ✅ Mark as READY — user can now query!
        if doc_id in _doc_registry:
            _doc_registry[doc_id].status = "ready"
            _doc_registry[doc_id].progress = 100.0
            _doc_registry[doc_id].chunk_count = len(chunks)
            logger.info(f"✅ {filename} is READY — {len(chunks)} chunks indexed in Qdrant.")
        else:
            logger.info(f"⚠️ Document {doc_id} was deleted during background processing.")

    except Exception as e:
        # ❌ Mark as FAILED with error message
        if doc_id in _doc_registry:
            _doc_registry[doc_id].status = "failed"
            _doc_registry[doc_id].progress = 0.0
            _doc_registry[doc_id].error = str(e)
        logger.error(f"❌ Background processing failed for {filename}: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ==========================================
# ENDPOINTS
# ==========================================
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    doc_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DIR, f"{doc_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Register document as "processing" immediately
    _doc_registry[doc_id] = DocumentStatus(
        doc_id=doc_id,
        filename=file.filename,
        status="processing",
        progress=5.0
    )

    # Fire off background embedding
    background_tasks.add_task(process_document_background, file_path, file.filename, doc_id)

    # Return instantly — frontend uses /status/{doc_id} to poll readiness
    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        status="processing"
    )


@app.get("/status/{doc_id}", response_model=DocumentStatus)
async def get_document_status(doc_id: str):
    """
    Poll this endpoint after upload to know when the document is ready to query.
    Status flow: processing → ready (or failed)
    """
    if doc_id not in _doc_registry:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return _doc_registry[doc_id]


@app.get("/documents", response_model=list[DocumentStatus])
async def list_documents():
    """Returns all documents and their current processing status."""
    return list(_doc_registry.values())

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in _doc_registry:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    
    try:
        services = _get_services()
        from qdrant_client.http import models
        client = services["qdrant"].client
        
        # Delete from Qdrant vector database using metadata filter
        client.delete(
            collection_name=services["qdrant"].collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id)
                    )
                ]
            )
        )
        
        # Remove from local registry
        del _doc_registry[doc_id]
        
        return {"status": "success", "message": f"Document {doc_id} deleted."}
    except Exception as e:
        logger.error(f"Failed to delete {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        services = _get_services()
        session_id = request.session_id or str(uuid.uuid4())

        # Ignite the LangGraph State Machine
        final_state = services["workflow"].run(request.question, session_id, request.search_mode, request.doc_id)

        # Format and return response
        context_chunks = final_state["retrieved_context"]
        sources = list(set([
            getattr(c.metadata, 'page', 0) if hasattr(c.metadata, 'page') else 0
            for c in context_chunks
        ]))

        return QueryResponse(
            answer=final_state["final_answer"],
            sources=sources,
            context_used=context_chunks
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "A.R.C.H.E.R. is online",
        "documents_indexed": len([d for d in _doc_registry.values() if d.status == "ready"])
    }
