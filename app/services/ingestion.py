import uuid
import os
from typing import List
import fitz  # PyMuPDF fallback
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from app.models.schemas import ChunkSchema, ChunkMetadata, ContentType
import logging

logger = logging.getLogger("A.R.C.H.E.R.Ingestion")


class IngestionService:
    def __init__(self):
        # 300-350 words ≈ 400-460 tokens (approx 1.33 tokens per word)
        # 50 words overlap ≈ 65 tokens
        self.chunk_size_tokens = 450
        self.chunk_overlap_tokens = 65
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size_tokens,
            chunk_overlap=self.chunk_overlap_tokens
        )

    def _detect_tables_in_text(self, text: str) -> bool:
        """Basic heuristic to detect if a chunk likely contains a table."""
        lines = text.strip().split("\n")
        pipe_lines = sum(1 for line in lines if "|" in line and line.count("|") >= 2)
        return pipe_lines >= 3

    def process_document(self, file_path: str, filename: str, doc_id: str = None) -> List[ChunkSchema]:
        """
        Parses a document (PDF or DOCX) page-by-page/element-by-element.
        Uses unstructured parser with robust fallbacks to python-docx and pdfplumber.
        """
        doc_id = doc_id or str(uuid.uuid4())
        logger.info(f"📄 Processing: {filename} (doc_id: {doc_id})")
        
        chunks = []
        chunk_index = 0
        ext = os.path.splitext(filename)[1].lower()

        # Try using unstructured library first
        unstructured_success = False
        try:
            from unstructured.partition.auto import partition
            logger.info("Parsing document using unstructured auto-partitioner...")
            elements = partition(filename=file_path)
            
            if elements:
                # Group elements by page (if available) or create virtual pages for docx
                pages = {}
                for el in elements:
                    # Get page number from unstructured metadata (default to 1)
                    page_num = getattr(el.metadata, "page_number", 1) or 1
                    if page_num not in pages:
                        pages[page_num] = []
                    pages[page_num].append(el.text)
                    
                for page_num, text_list in sorted(pages.items()):
                    page_text = "\n".join(text_list).strip()
                    if not page_text:
                        continue
                        
                    page_doc = Document(
                        text=page_text,
                        metadata={
                            "filename": filename,
                            "doc_id": doc_id,
                            "page": page_num
                        }
                    )
                    nodes = self.splitter.get_nodes_from_documents([page_doc])
                    for node in nodes:
                        is_table = self._detect_tables_in_text(node.text)
                        content_type = ContentType.TABLE if is_table else ContentType.TEXT
                        chunks.append(ChunkSchema(
                            chunk_id=f"{doc_id}_chunk_{chunk_index}",
                            text=node.text.strip(),
                            metadata=ChunkMetadata(
                                doc_id=doc_id,
                                filename=filename,
                                page=page_num,
                                content_type=content_type,
                                section_heading="General"
                            )
                        ))
                        chunk_index += 1
                unstructured_success = True
                logger.info("Unstructured partitioning completed successfully.")
            else:
                logger.warning("Unstructured auto-partitioner returned 0 elements.")
        except Exception as e:
            logger.warning(f"⚠️ Unstructured partitioning failed or not fully configured: {e}. Falling back to native parsers.")

        if not unstructured_success:
            # Fallbacks: PDF -> pdfplumber, DOCX -> python-docx
            if ext == ".pdf":
                import pdfplumber
                logger.info("Using pdfplumber fallback for PDF text extraction...")
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        text = text.strip()
                        page.flush_cache()  # Crucial: release page layout resources from memory
                        if not text:
                            continue
                        page_doc = Document(
                            text=text,
                            metadata={
                                "filename": filename, 
                                "doc_id": doc_id, 
                                "page": page_num + 1
                            }
                        )
                        nodes = self.splitter.get_nodes_from_documents([page_doc])
                        for node in nodes:
                            is_table = self._detect_tables_in_text(node.text)
                            content_type = ContentType.TABLE if is_table else ContentType.TEXT
                            chunks.append(ChunkSchema(
                                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                                text=node.text.strip(),
                                metadata=ChunkMetadata(
                                    doc_id=doc_id,
                                    filename=filename,
                                    page=page_num + 1,
                                    content_type=content_type,
                                    section_heading="General"
                                )
                            ))
                            chunk_index += 1
            elif ext in [".docx", ".doc"]:
                import docx
                logger.info("Using python-docx fallback for Word text extraction...")
                doc = docx.Document(file_path)
                # Word files don't have hard pages, group paragraphs into batches as "virtual pages"
                current_text_batch = []
                virtual_page = 1
                for para in doc.paragraphs:
                    if para.text.strip():
                        current_text_batch.append(para.text.strip())
                    if len(current_text_batch) >= 15:  # Every 15 paragraphs is a virtual page
                        page_text = "\n".join(current_text_batch)
                        page_doc = Document(
                            text=page_text,
                            metadata={
                                "filename": filename,
                                "doc_id": doc_id,
                                "page": virtual_page
                            }
                        )
                        nodes = self.splitter.get_nodes_from_documents([page_doc])
                        for node in nodes:
                            is_table = self._detect_tables_in_text(node.text)
                            content_type = ContentType.TABLE if is_table else ContentType.TEXT
                            chunks.append(ChunkSchema(
                                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                                text=node.text.strip(),
                                metadata=ChunkMetadata(
                                    doc_id=doc_id,
                                    filename=filename,
                                    page=virtual_page,
                                    content_type=content_type,
                                    section_heading="General"
                                )
                            ))
                            chunk_index += 1
                        current_text_batch = []
                        virtual_page += 1
                # Process remaining paragraphs
                if current_text_batch:
                    page_text = "\n".join(current_text_batch)
                    page_doc = Document(
                        text=page_text,
                        metadata={
                            "filename": filename,
                            "doc_id": doc_id,
                            "page": virtual_page
                        }
                    )
                    nodes = self.splitter.get_nodes_from_documents([page_doc])
                    for node in nodes:
                        is_table = self._detect_tables_in_text(node.text)
                        content_type = ContentType.TABLE if is_table else ContentType.TEXT
                        chunks.append(ChunkSchema(
                            chunk_id=f"{doc_id}_chunk_{chunk_index}",
                            text=node.text.strip(),
                            metadata=ChunkMetadata(
                                doc_id=doc_id,
                                filename=filename,
                                page=virtual_page,
                                content_type=content_type,
                                section_heading="General"
                            )
                        ))
                        chunk_index += 1
            else:
                logger.error(f"Unsupported file format: {ext}")
                raise ValueError(f"Unsupported file extension: {ext}")

        if not chunks:
            logger.error(f"Failed to extract any text chunks from: {filename}")
            raise ValueError("No text could be extracted from this document. It may be empty or require OCR.")

        # Explicitly run garbage collection to reclaim memory from heavy parser sessions
        import gc
        gc.collect()

        logger.info(f"  ✅ Ingestion complete: {len(chunks)} chunks created for {filename}")
        return chunks

