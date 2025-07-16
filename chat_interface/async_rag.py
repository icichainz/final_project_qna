# Dependencies (install via pip)
# pip install langchain faiss-cpu pdfplumber numpy chainlit aiofiles sentence-transformers torch

import os
import glob
import faiss
import pickle
import asyncio
import aiofiles
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import logging
from enum import Enum
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import torch

# PDF processing libraries
import pdfplumber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Configuration for async processing
MAX_CONCURRENT_PDFS = 3
MAX_CONCURRENT_CHUNKS = 100
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CACHE_FOLDER = "txt_cache"

class PDFParsingMethod(Enum):
    """Enumeration of available PDF parsing methods."""
    PDFPLUMBER = "pdfplumber"

@dataclass
class PDFParsingConfig:
    """Configuration for PDF parsing methods."""
    method: PDFParsingMethod = PDFParsingMethod.PDFPLUMBER
    extract_tables: bool = True
    preserve_formatting: bool = True
    min_text_length: int = 100

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = EMBEDDING_MODEL
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    cache_folder: str = "embedding_cache"

class AsyncPDFRAG:
    def __init__(self, max_concurrent_pdfs: int = MAX_CONCURRENT_PDFS,
                 max_concurrent_chunks: int = MAX_CONCURRENT_CHUNKS,
                 cache_folder: str = CACHE_FOLDER,
                 parsing_config: PDFParsingConfig = None,
                 embedding_config: EmbeddingConfig = None):
        self.max_concurrent_pdfs = max_concurrent_pdfs
        self.max_concurrent_chunks = max_concurrent_chunks
        self.cache_folder = cache_folder
        self.parsing_config = parsing_config or PDFParsingConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()

        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_pdfs)

        # Initialize embedding model
        self._init_embedding_model()

        # Create cache folders
        os.makedirs(self.cache_folder, exist_ok=True)
        os.makedirs(self.embedding_config.cache_folder, exist_ok=True)

    def _init_embedding_model(self):
        """Initialize the sentence transformer model."""
        try:
            # Determine device
            if self.embedding_config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.embedding_config.device

            logger.info(f"Initializing embedding model: {self.embedding_config.model_name} on {device}")

            # Initialize model
            self.embedding_model = SentenceTransformer(
                self.embedding_config.model_name,
                device=device,
                cache_folder=self.embedding_config.cache_folder
            )

            # Set max sequence length if specified
            if self.embedding_config.max_seq_length:
                self.embedding_model.max_seq_length = self.embedding_config.max_seq_length

            logger.info(f"Embedding model initialized successfully on {device}")

        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise

    def get_cache_path(self, pdf_path: str) -> str:
        """Generate cache file path for a given PDF."""
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        return os.path.join(self.cache_folder, f"{pdf_name}_pdfplumber.txt")

    async def is_pdf_cached(self, pdf_path: str) -> bool:
        """Check if PDF has already been parsed and cached."""
        cache_path = self.get_cache_path(pdf_path)
        if not os.path.exists(cache_path):
            return False

        try:
            pdf_mtime = os.path.getmtime(pdf_path)
            cache_mtime = os.path.getmtime(cache_path)
            return cache_mtime > pdf_mtime
        except OSError:
            return False

    async def load_from_cache(self, pdf_path: str) -> Optional[str]:
        """Load parsed text from cache if available."""
        cache_path = self.get_cache_path(pdf_path)
        try:
            async with aiofiles.open(cache_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                logger.info(f"Loaded cached text for {os.path.basename(pdf_path)}")
                return content
        except Exception as e:
            logger.error(f"Error loading cache for {pdf_path}: {e}")
            return None

    async def save_to_cache(self, pdf_path: str, content: str):
        """Save parsed text to cache."""
        cache_path = self.get_cache_path(pdf_path)
        try:
            async with aiofiles.open(cache_path, 'w', encoding='utf-8') as f:
                await f.write(content)
                logger.info(f"Cached parsed text for {os.path.basename(pdf_path)}")
        except Exception as e:
            logger.error(f"Error saving cache for {pdf_path}: {e}")

    def parse_pdf_with_pdfplumber(self, pdf_path: str) -> str:
        """Parse PDF using pdfplumber library."""
        try:
            text_content = []

            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                metadata = pdf.metadata or {}
                if metadata:
                    text_content.append("=== DOCUMENT METADATA ===")
                    for key, value in metadata.items():
                        if value:
                            text_content.append(f"{key}: {value}")
                    text_content.append("")

                # Extract text from each page
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_content.append(f"=== PAGE {page_num} ===")
                        text_content.append(page_text.strip())
                        text_content.append("")

                    # Extract tables if configured
                    if self.parsing_config.extract_tables:
                        tables = page.extract_tables()
                        for table_num, table in enumerate(tables, 1):
                            text_content.append(f"--- Table {table_num} on Page {page_num} ---")
                            for row in table:
                                if row:
                                    text_content.append(" | ".join(str(cell) if cell else "" for cell in row))
                            text_content.append("")

            return "\n".join(text_content)

        except Exception as e:
            logger.error(f"Error parsing PDF with pdfplumber {pdf_path}: {e}")
            return ""

    def evaluate_parsing_quality(self, text: str, pdf_path: str) -> Dict[str, Any]:
        """Evaluate the quality of parsed text."""
        if not text:
            return {"score": 0, "reason": "No text extracted"}

        text_length = len(text.strip())
        word_count = len(text.split())
        line_count = len(text.split('\n'))

        has_structure = any(marker in text for marker in ['===', '---', '#', '##'])
        has_metadata = 'METADATA' in text or 'metadata' in text.lower()

        score = 0
        reasons = []

        if text_length >= self.parsing_config.min_text_length:
            score += 40
        else:
            reasons.append(f"Text too short ({text_length} chars)")

        if word_count > 50:
            score += 20

        if has_structure:
            score += 20

        if has_metadata:
            score += 10

        if line_count > 10:
            score += 10

        return {
            "score": score,
            "text_length": text_length,
            "word_count": word_count,
            "line_count": line_count,
            "has_structure": has_structure,
            "has_metadata": has_metadata,
            "reasons": reasons
        }

    async def parse_pdf(self, pdf_path: str) -> str:
        """Parse PDF using pdfplumber with caching."""
        # Check cache first
        if await self.is_pdf_cached(pdf_path):
            cached_content = await self.load_from_cache(pdf_path)
            if cached_content:
                return cached_content

        logger.info(f"Parsing {os.path.basename(pdf_path)} with pdfplumber")

        # Parse with pdfplumber
        loop = asyncio.get_event_loop()
        parsed_text = await loop.run_in_executor(self.executor, self.parse_pdf_with_pdfplumber, pdf_path)

        # Cache the result if parsing was successful
        if parsed_text:
            await self.save_to_cache(pdf_path, parsed_text)

        return parsed_text

    def chunk_text(self, full_text: str) -> List[str]:
        """Split text into semantically coherent chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        return splitter.split_text(full_text)

    async def embed_chunks_batch(self, chunks: List[str]) -> np.ndarray:
        """Embed chunks using all-mpnet-base-v2 sentence transformer model."""
        if not chunks:
            return np.array([])

        def _embed_batch(chunk_batch):
            try:
                # Encode chunks using sentence transformer
                embeddings = self.embedding_model.encode(
                    chunk_batch,
                    batch_size=self.embedding_config.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.embedding_config.normalize_embeddings
                )
                return embeddings
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                return np.array([])

        # Process in batches to manage memory
        batch_size = self.embedding_config.batch_size
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]

        logger.info(f"Embedding {len(chunks)} chunks in {len(batches)} batches using {self.embedding_config.model_name}")

        # Run batches concurrently
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(self.executor, _embed_batch, batch) for batch in batches]

        embeddings_batches = await asyncio.gather(*tasks)

        # Concatenate all embeddings
        all_embeddings = []
        for batch_embeddings in embeddings_batches:
            if len(batch_embeddings) > 0:
                all_embeddings.append(batch_embeddings)

        if all_embeddings:
            return np.vstack(all_embeddings).astype("float32")
        else:
            return np.array([])

    async def build_faiss_index(self, chunks: List[str]) -> Tuple[faiss.Index, List[str]]:
        """Build FAISS index asynchronously."""
        if not chunks:
            raise ValueError("No chunks provided for indexing")

        embeddings_np = await self.embed_chunks_batch(chunks)

        if len(embeddings_np) == 0:
            raise ValueError("Failed to generate embeddings")

        # Build the FAISS index
        dim = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings_np)

        logger.info(f"FAISS index built successfully with {len(chunks)} chunks, embedding dim: {dim}")
        return index, chunks.copy()

    async def save_index(self, index: faiss.Index, metadata: List[str],
                        index_path: str, meta_path: str):
        """Save FAISS index and metadata asynchronously."""
        def _save_index():
            faiss.write_index(index, index_path)

        def _save_metadata():
            with open(meta_path, "wb") as f:
                pickle.dump(metadata, f)

        loop = asyncio.get_event_loop()
        await asyncio.gather(
            loop.run_in_executor(self.executor, _save_index),
            loop.run_in_executor(self.executor, _save_metadata)
        )

        logger.info(f"Index saved to {index_path} and metadata to {meta_path}")

    async def load_index(self, index_path: str, meta_path: str) -> Tuple[faiss.Index, List[str]]:
        """Load FAISS index and metadata asynchronously."""
        def _load_index():
            return faiss.read_index(index_path)

        def _load_metadata():
            with open(meta_path, "rb") as f:
                return pickle.load(f)

        loop = asyncio.get_event_loop()
        index, metadata = await asyncio.gather(
            loop.run_in_executor(self.executor, _load_index),
            loop.run_in_executor(self.executor, _load_metadata)
        )

        logger.info("Index and metadata loaded successfully.")
        return index, metadata

    async def query_index(self, query: str, index: faiss.Index,
                         metadata: List[str], top_k: int = 5) -> List[str]:
        """Query the index asynchronously using all-mpnet-base-v2 embedding model."""
        def _embed_query():
            return self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=self.embedding_config.normalize_embeddings
            )[0]

        def _search_index(q_emb):
            distances, indices = index.search(np.array([q_emb], dtype="float32"), top_k)
            return [metadata[i] for i in indices[0]]

        loop = asyncio.get_event_loop()
        q_emb = await loop.run_in_executor(self.executor, _embed_query)
        results = await loop.run_in_executor(self.executor, _search_index, q_emb)

        return results

    def select_pdf_files(self, pdf_files: List[str], limit: Optional[int] = None,
                        offset: int = 0) -> List[str]:
        """Select a subset of PDF files based on offset and limit."""
        total_files = len(pdf_files)

        if offset >= total_files:
            logger.warning(f"Offset {offset} is >= total files {total_files}. No files selected.")
            return []

        sorted_files = sorted(pdf_files)
        selected_files = sorted_files[offset:]

        if limit is not None:
            selected_files = selected_files[:limit]

        logger.info(f"Selected {len(selected_files)} files (offset: {offset}, limit: {limit}) from {total_files} total files")

        for i, file in enumerate(selected_files):
            logger.info(f"  {i+1}. {os.path.basename(file)}")

        return selected_files

    async def process_pdf_batch(self, pdf_files: List[str]) -> List[str]:
        """Process multiple PDFs concurrently using pdfplumber."""
        semaphore = asyncio.Semaphore(self.max_concurrent_pdfs)

        async def _process_single_pdf(pdf_file):
            async with semaphore:
                structured_text = await self.parse_pdf(pdf_file)
                if structured_text:
                    return self.chunk_text(structured_text)
                return []

        tasks = [_process_single_pdf(pdf_file) for pdf_file in pdf_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_chunks = []
        processed_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing PDF {pdf_files[i]}: {result}")
            elif result:
                all_chunks.extend(result)
                processed_count += 1

        logger.info(f"Successfully processed {processed_count} PDFs")
        return all_chunks

    async def ingest_and_index(self, pdf_folder: str, index_path: str, meta_path: str,
                              limit: Optional[int] = None, offset: int = 0):
        """Complete ingestion pipeline using pdfplumber and all-mpnet-base-v2."""
        all_pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
        if not all_pdf_files:
            logger.warning(f"No PDF files found in '{pdf_folder}'.")
            return

        pdf_files = self.select_pdf_files(all_pdf_files, limit=limit, offset=offset)

        if not pdf_files:
            logger.warning("No PDF files selected for processing.")
            return

        logger.info(f"Processing {len(pdf_files)} PDF files with pdfplumber")
        logger.info(f"Using embedding model: {self.embedding_config.model_name}")

        all_chunks = await self.process_pdf_batch(pdf_files)

        if all_chunks:
            logger.info(f"Generated {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
            index, metadata = await self.build_faiss_index(all_chunks)
            await self.save_index(index, metadata, index_path, meta_path)
        else:
            logger.warning("No text was extracted from the PDFs. Index not built.")

    async def clear_cache(self):
        """Clear all cached text files."""
        try:
            import shutil
            if os.path.exists(self.cache_folder):
                shutil.rmtree(self.cache_folder)
                os.makedirs(self.cache_folder, exist_ok=True)
                logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    async def get_cache_stats(self) -> dict:
        """Get statistics about cached files."""
        try:
            cache_files = glob.glob(os.path.join(self.cache_folder, "*.txt"))
            total_size = sum(os.path.getsize(f) for f in cache_files)

            return {
                "cached_files": len(cache_files),
                "total_size_mb": total_size / (1024 * 1024),
                "cache_folder": self.cache_folder,
                "embedding_model": self.embedding_config.model_name,
                "embedding_device": str(self.embedding_model.device),
                "parsing_method": "pdfplumber"
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

    def __del__(self):
        """Cleanup executor on destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# --- Example Usage ---
async def main():
    # Configuration
    PDF_FOLDER_PATH = '/content/pdfs_20'
    INDEX_FILE_PATH = 'my_document_index.faiss'
    METADATA_FILE_PATH = 'my_document_meta.pkl'
    CACHE_FOLDER = 'txt_cache'

    # Test configuration
    TEST_LIMIT = 1
    TEST_OFFSET = 0

    # Configure parsing and embedding
    parsing_config = PDFParsingConfig(
        method=PDFParsingMethod.PDFPLUMBER,
        extract_tables=True,
        preserve_formatting=True,
        min_text_length=100
    )

    embedding_config = EmbeddingConfig(
        model_name=EMBEDDING_MODEL,
        device="auto",
        batch_size=32,
        normalize_embeddings=True
    )

    # Create the PDF folder if it doesn't exist
    if not os.path.exists(PDF_FOLDER_PATH):
        os.makedirs(PDF_FOLDER_PATH)
        logger.info(f"Created folder '{PDF_FOLDER_PATH}'. Please add your PDF files there.")
        return

    # Initialize the RAG system
    rag_system = AsyncPDFRAG(
        cache_folder=CACHE_FOLDER,
        parsing_config=parsing_config,
        embedding_config=embedding_config
    )

    try:
        # Show cache statistics
        cache_stats = await rag_system.get_cache_stats()
        logger.info(f"Cache stats: {cache_stats}")

        # Run the ingestion and indexing pipeline
        logger.info(f"Testing with {TEST_LIMIT} documents starting from offset {TEST_OFFSET}")
        logger.info(f"Using pdfplumber for PDF parsing")
        logger.info(f"Using {EMBEDDING_MODEL} for embeddings")

        await rag_system.ingest_and_index(
            PDF_FOLDER_PATH,
            INDEX_FILE_PATH,
            METADATA_FILE_PATH,
            limit=TEST_LIMIT,
            offset=TEST_OFFSET
        )

        # Show updated cache statistics
        cache_stats = await rag_system.get_cache_stats()
        logger.info(f"Updated cache stats: {cache_stats}")

        # Example query after indexing
        if os.path.exists(INDEX_FILE_PATH):
            index, metadata = await rag_system.load_index(INDEX_FILE_PATH, METADATA_FILE_PATH)

            # Test query
            test_query = "What is the main topic of the documents?"
            results = await rag_system.query_index(test_query, index, metadata)

            logger.info(f"Query: {test_query}")
            logger.info(f"Results: {len(results)} chunks retrieved")

            # Show sample results
            for i, result in enumerate(results[:2]):
                logger.info(f"Result {i+1}: {result[:200]}...")

    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")

# --- Performance Testing ---
async def performance_test():
    """Test performance with different batch sizes and configurations."""

    configs = [
        {"batch_size": 16, "max_concurrent": 2},
        {"batch_size": 32, "max_concurrent": 3},
        {"batch_size": 64, "max_concurrent": 4}
    ]

    for config in configs:
        logger.info(f"\n=== Testing config: {config} ===")

        embedding_config = EmbeddingConfig(
            model_name=EMBEDDING_MODEL,
            batch_size=config["batch_size"]
        )

        rag_system = AsyncPDFRAG(
            max_concurrent_pdfs=config["max_concurrent"],
            cache_folder=f'test_cache_{config["batch_size"]}',
            embedding_config=embedding_config
        )

        start_time = asyncio.get_event_loop().time()

        try:
            await rag_system.ingest_and_index(
                '/content/pdfs_20',
                f'test_index_{config["batch_size"]}.faiss',
                f'test_meta_{config["batch_size"]}.pkl',
                limit=5,
                offset=0
            )

            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time

            logger.info(f"Config {config} - Total time: {total_time:.2f}s")

        except Exception as e:
            logger.error(f"Error in performance test: {e}")

if __name__ == '__main__':
    # Check if we're in a notebook environment (Jupyter/Colab)
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we get here, we're in a notebook with an existing event loop
        import nest_asyncio
        nest_asyncio.apply()  # Allow nested event loops

        # Run main example
        asyncio.run(main())

        # Uncomment to run performance test:
        # asyncio.run(performance_test())

    except RuntimeError:
        # No event loop running, we can use asyncio.run() normally
        asyncio.run(main())
    except ImportError:
        # nest_asyncio not available, use alternative approach
        try:
            # Get the current event loop
            loop = asyncio.get_event_loop()
            # Run the main function
            loop.run_until_complete(main())
        except RuntimeError:
            # Create a new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(main())
            loop.close()
