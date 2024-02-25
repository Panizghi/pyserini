

'''This class manages the DuckDB connection, 
   creating an embeddings table, 
   and methods for inserting and querying embeddings. '''


import duckdb
import numpy as np

class DuckDBEmbeddings:
    def __init__(self, db_path=':memory:'):
        """Initialize DuckDB connection, optionally as an in-memory database."""
        self.conn = duckdb.connect(database=db_path, read_only=False)
        self.setup_embeddings_table()
    
    def setup_embeddings_table(self):
        """Create the embeddings table if it doesn't exist."""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id VARCHAR PRIMARY KEY,
                embedding BLOB
            )
        ''')
    
    def insert_embedding(self, doc_id, embedding):
        """Insert a single embedding into the database."""
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        self.conn.execute('INSERT INTO embeddings VALUES (?, ?)', (doc_id, embedding.tobytes()))
    
    def get_embeddings(self, doc_ids):
        """Retrieve embeddings for a list of document IDs."""
        placeholders = ','.join(['?']*len(doc_ids))
        result = self.conn.execute(f'SELECT doc_id, embedding FROM embeddings WHERE doc_id IN ({placeholders})', doc_ids).fetchall()
        return {doc_id: np.frombuffer(embedding, dtype=np.float32) for doc_id, embedding in result}


'''Functions to insert embeddings from an `.npz` 
  file or a Python dictionary into DuckDB: '''


def insert_embeddings_from_npz(duckdb_embeddings, npz_path):
    """Load embeddings from an .npz file and insert into DuckDB."""
    embeddings = np.load(npz_path, allow_pickle=True)
    for doc_id in embeddings.files:
        embedding = embeddings[doc_id]
        duckdb_embeddings.insert_embedding(doc_id, embedding)

def insert_embeddings_from_dict(duckdb_embeddings, embeddings_dict):
    """Insert embeddings from a dictionary into DuckDB."""
    for doc_id, embedding in embeddings_dict.items():
        duckdb_embeddings.insert_embedding(doc_id, embedding)

'''Integrates DuckDB into Pyserini's search process, utilizing embeddings for second-stage retrieval'''

from pyserini.search.lucene import LuceneImpactSearcher
from scipy.spatial.distance import cdist

class DuckDBLuceneSearcher(LuceneImpactSearcher):
    def __init__(self, index_dir, duckdb_path=':memory:', query_encoder=None):
        """Initialize with Lucene index, DuckDB path, and query encoder."""
        super().__init__(index_dir)
        self.duckdb_embeddings = DuckDBEmbeddings(duckdb_path)
        self.query_encoder = query_encoder
    
    def search_with_embeddings(self, query, k=10):
        """Perform search with embeddings, including initial Lucene search and second-stage reranking."""
        initial_hits = super().search(query, k)
        doc_ids = [hit.docid for hit in initial_hits]
        doc_embeddings = self.duckdb_embeddings.get_embeddings(doc_ids)
        
        query_embedding = self.query_encoder.encode(query)
        doc_embeddings_array = np.array([doc_embeddings[doc_id] for doc_id in doc_ids])
        similarities = 1 - cdist([query_embedding], doc_embeddings_array, metric='cosine')[0]
        
        ranked_doc_ids = [doc for _, doc in sorted(zip(similarities, doc_ids), reverse=True)]
        return [hit for doc_id in ranked_doc_ids for hit in initial_hits if hit.docid == doc_id]

