from neo4j import GraphDatabase
from app.core.config import settings
import logging

logger = logging.getLogger("A.R.C.H.E.R.Neo4j")

class Neo4jManager:
    def __init__(self):
        self.uri = settings.NEO4J_URI
        self.username = settings.NEO4J_USERNAME
        self.password = settings.NEO4J_PASSWORD
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connectivity
            self.driver.verify_connectivity()
            logger.info("🔌 Connected to Neo4j database service successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j at {self.uri}: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def execute_query(self, query: str, parameters: dict = None):
        if not self.driver:
            logger.warning("Neo4j driver is not initialized.")
            return []
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Neo4j query execution failed: {e}")
            raise e

    def create_chunk_node(self, chunk_id: str, text: str, doc_id: str, page: int, filename: str):
        """Creates a Chunk node in Neo4j."""
        query = (
            "MERGE (c:Chunk {chunk_id: $chunk_id}) "
            "SET c.text = $text, c.doc_id = $doc_id, c.page = $page, c.filename = $filename "
            "RETURN c"
        )
        parameters = {
            "chunk_id": chunk_id,
            "text": text,
            "doc_id": doc_id,
            "page": page,
            "filename": filename
        }
        self.execute_query(query, parameters)

    def add_extracted_relation(self, source: str, target: str, relationship: str, chunk_id: str, doc_id: str):
        """Creates Entity nodes, RELATED_TO relationships, and MENTIONED_IN relationships linking to Chunks."""
        # 1. Merge entities and relate them
        query_relation = (
            "MERGE (s:Entity {name: toLower($source)}) "
            "ON CREATE SET s.name_raw = $source, s.doc_id = $doc_id "
            "MERGE (t:Entity {name: toLower($target)}) "
            "ON CREATE SET t.name_raw = $target, t.doc_id = $doc_id "
            "MERGE (s)-[r:RELATED_TO {type: toLower($relationship)}]->(t) "
            "ON CREATE SET r.doc_id = $doc_id, r.type_raw = $relationship "
            "RETURN s, t"
        )
        self.execute_query(query_relation, {
            "source": source.strip(),
            "target": target.strip(),
            "relationship": relationship.strip(),
            "doc_id": doc_id
        })

        # 2. Link source and target entities to the Chunk where they were mentioned
        query_mention = (
            "MATCH (c:Chunk {chunk_id: $chunk_id}) "
            "MATCH (s:Entity {name: toLower($source)}) "
            "MATCH (t:Entity {name: toLower($target)}) "
            "MERGE (s)-[:MENTIONED_IN {doc_id: $doc_id}]->(c) "
            "MERGE (t)-[:MENTIONED_IN {doc_id: $doc_id}]->(c)"
        )
        self.execute_query(query_mention, {
            "chunk_id": chunk_id,
            "source": source.strip(),
            "target": target.strip(),
            "doc_id": doc_id
        })

    def delete_document_nodes(self, doc_id: str):
        """Transactionally purges all Chunk nodes, Entity nodes, and relationships matching the doc_id."""
        logger.info(f"🧹 Purging Neo4j graph nodes and relationships for doc_id: {doc_id}")
        
        # Delete relationships having doc_id
        self.execute_query(
            "MATCH ()-[r {doc_id: $doc_id}]->() DELETE r",
            {"doc_id": doc_id}
        )
        
        # Delete Chunk nodes with doc_id
        self.execute_query(
            "MATCH (c:Chunk {doc_id: $doc_id}) DETACH DELETE c",
            {"doc_id": doc_id}
        )
        
        # Delete Entity nodes tagged with doc_id
        self.execute_query(
            "MATCH (e:Entity {doc_id: $doc_id}) DETACH DELETE e",
            {"doc_id": doc_id}
        )
        
        # Clean up any orphan Entity nodes (entities no longer mentioned in any Chunk)
        self.execute_query(
            "MATCH (e:Entity) WHERE NOT (e)-[:MENTIONED_IN]->() DETACH DELETE e"
        )
        
        logger.info(f"✅ Successfully cleaned up Neo4j graph nodes/relations for doc_id: {doc_id}")
