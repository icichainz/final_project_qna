import os
import shutil
import tempfile
import asyncio
from typing import List, Optional

import chainlit as cl
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding & indexing
import glob
import faiss
import pickle
import aiofiles
import pdfplumber
from sentence_transformers import SentenceTransformer
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-mpnet-base-v2"
CACHE_FOLDER = "txt_cache"
EMBED_CACHE = "emb_cache"
INDEX_FILE = "pdf_index.faiss"
META_FILE = "pdf_meta.pkl"
MAX_PDFS = 3

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# PDF parsing config
class PDFParsingMethod(Enum):
    PDFPLUMBER = "pdfplumber"

@dataclass
class PDFParsingConfig:
    method: PDFParsingMethod = PDFParsingMethod.PDFPLUMBER
    extract_tables: bool = True
    preserve_formatting: bool = True
    min_text_length: int = 100

@dataclass
class EmbeddingConfig:
    model_name: str = EMBEDDING_MODEL
    device: str = "auto"
    batch_size: int = 32
    normalize: bool = True
    cache_folder: str = EMBED_CACHE

class AsyncPDFRAG:
    def __init__(self,
                 cache_folder: str = CACHE_FOLDER,
                 parsing_config: PDFParsingConfig = None,
                 embedding_config: EmbeddingConfig = None,
                 max_workers: int = MAX_PDFS):
        self.cache_folder = cache_folder
        os.makedirs(cache_folder, exist_ok=True)
        self.parsing_config = parsing_config or PDFParsingConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        os.makedirs(self.embedding_config.cache_folder, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # Init model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(
            self.embedding_config.model_name,
            device=device,
            cache_folder=self.embedding_config.cache_folder
        )
        self.model.max_seq_length = self.embedding_config.model_name and 512

    def get_text_cache_path(self, pdf_path: str) -> str:
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        return os.path.join(self.cache_folder, f"{name}.txt")

    async def load_cached_text(self, pdf_path: str) -> Optional[str]:
        path = self.get_text_cache_path(pdf_path)
        if os.path.exists(path):
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                return await f.read()
        return None

    async def save_cached_text(self, pdf_path: str, text: str):
        path = self.get_text_cache_path(pdf_path)
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            await f.write(text)

    def parse_pdf(self, pdf_path: str) -> str:
        text_chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            # metadata
            meta = pdf.metadata or {}
            if meta:
                text_chunks.append("=== METADATA ===")
                for k, v in meta.items():
                    if v:
                        text_chunks.append(f"{k}: {v}")
                text_chunks.append("")
            # pages
            for i, page in enumerate(pdf.pages, 1):
                txt = page.extract_text() or ""
                if txt.strip():
                    text_chunks.append(f"=== PAGE {i} ===")
                    text_chunks.append(txt.strip())
                    text_chunks.append("")
                if self.parsing_config.extract_tables:
                    for t_i, table in enumerate(page.extract_tables(), 1):
                        text_chunks.append(f"--- TABLE {t_i} on PAGE {i} ---")
                        for row in table:
                            text_chunks.append(" | ".join(str(c or '') for c in row))
                        text_chunks.append("")
        return "\n".join(text_chunks)

    async def get_full_text(self, pdf_path: str) -> str:
        # Try cache
        cached = await self.load_cached_text(pdf_path)
        if cached:
            return cached
        # Parse and cache
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(self.executor, self.parse_pdf, pdf_path)
        await self.save_cached_text(pdf_path, text)
        return text

    def chunk_text(self, text: str) -> List[str]:
        return text_splitter.split_text(text)

    async def build_index(self, chunks: List[str]):
        # embed
        def embed_batch(batch):
            return self.model.encode(
                batch,
                batch_size=self.embedding_config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.embedding_config.normalize
            )
        loop = asyncio.get_event_loop()
        embeddings = []
        for i in range(0, len(chunks), self.embedding_config.batch_size):
            batch = chunks[i:i+self.embedding_config.batch_size]
            em = await loop.run_in_executor(self.executor, embed_batch, batch)
            embeddings.append(em)
        ems = __import__('numpy').vstack(embeddings).astype('float32')
        # faiss
        dim = ems.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(ems)
        # save
        faiss.write_index(index, INDEX_FILE)
        with open(META_FILE, 'wb') as f:
            pickle.dump(chunks, f)
        return index, chunks

    async def load_index(self):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, 'rb') as f:
            meta = pickle.load(f)
        return index, meta

    async def query(self, query: str, top_k: int = 5) -> List[str]:
        index, meta = await self.load_index()
        # embed query
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=self.embedding_config.normalize)[0]
        D, I = index.search(__import__('numpy').array([q_emb], dtype='float32'), top_k)
        return [meta[i] for i in I[0]]

# Initialize RAG helper
rag = AsyncPDFRAG()

# Utility to save uploaded file temporarily

def save_temp_copy(uploaded_path: str) -> str:
    tempdir = tempfile.mkdtemp()
    dest = shutil.copy(uploaded_path, tempdir)
    return dest

@cl.on_chat_start
async def start():
    # Ask for file
    file_msg = await cl.AskFileMessage(
        content="Upload a PDF to ingest and index:",
        accept=["application/pdf"],
        max_size_mb=20
    ).send()
    file = file_msg[0]
    path = save_temp_copy(file.path)
    await cl.Message(content=f"Parsing and indexing `{file.name}`...").send()
    # Full text + chunks
    full_text = await rag.get_full_text(path)
    chunks = rag.chunk_text(full_text)
    # Build index if not exist
    if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
        await rag.build_index(chunks)
    await cl.Message(content="Indexing done. You can now ask questions.").send()
    # Save chain components
    history = ChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=history)
    # Simple retriever wrapper
    class Retriever:
        def __init__(self, rag_helper):
            self.rag = rag_helper
        async def get_relevant_documents(self, query: str):
            snippets = await self.rag.query(query, top_k=5)
            return [Document(page_content=snip) for snip in snippets]
    retriever = Retriever(rag)
    chain = ConversationalRetrievalChain.from_llm(
        ChatAnthropic(model="claude-3-5-sonnet-20240620"),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    ans = res["answer"]
    docs = res.get("source_documents", [])
    elements = [cl.Text(content=d.page_content, name=f"src{i}", display="side") for i, d in enumerate(docs)]
    if elements:
        ans += "\nSources: " + ", ".join(e.name for e in elements)
    await cl.Message(content=ans, elements=elements).send()
