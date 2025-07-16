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
        self.model.max_seq_length = 512
        # Load or init index
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self.index, self.meta = self._load_index()
        else:
            self.index = None
            self.meta = []

    def _load_index(self):
        idx = faiss.read_index(INDEX_FILE)
        with open(META_FILE, 'rb') as f:
            meta = pickle.load(f)
        return idx, meta

    def _save_index(self):
        faiss.write_index(self.index, INDEX_FILE)
        with open(META_FILE, 'wb') as f:
            pickle.dump(self.meta, f)

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
                            text_chunks.append(" | ".join(
                                str(c or '') for c in row))
                        text_chunks.append("")
        return "\n".join(text_chunks)

    async def get_full_text(self, pdf_path: str) -> str:
        cached = await self.load_cached_text(pdf_path)
        if cached:
            return cached
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(self.executor, self.parse_pdf, pdf_path)
        await self.save_cached_text(pdf_path, text)
        return text

    def chunk_text(self, text: str) -> List[str]:
        return text_splitter.split_text(text)

    def init_index(self, chunks: List[str]):
        ems = self.model.encode(
            chunks,
            batch_size=self.embedding_config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.embedding_config.normalize
        ).astype('float32')
        dim = ems.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(ems)
        self.meta = chunks.copy()
        self._save_index()

    def extend_index(self, new_chunks: List[str]):
        ems = self.model.encode(
            new_chunks,
            batch_size=self.embedding_config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.embedding_config.normalize
        ).astype('float32')
        self.index.add(ems)
        self.meta.extend(new_chunks)
        self._save_index()

    def query(self, query: str, top_k: int = 5) -> List[str]:
        q_emb = self.model.encode([query], convert_to_numpy=True,
                                  normalize_embeddings=self.embedding_config.normalize)[0]
        D, I = self.index.search(q_emb.reshape(1, -1), top_k)
        return [self.meta[i] for i in I[0]]


# Initialize RAG helper
rag = AsyncPDFRAG()


def save_temp_copy(uploaded_path: str) -> str:
    tempdir = tempfile.mkdtemp()
    return shutil.copy(uploaded_path, tempdir)


@cl.on_chat_start
async def start():
    # Ingest initial PDFs from a folder if exists
    pdfs = glob.glob("./pdfs/*.pdf")
    for pdf in pdfs:
        path = save_temp_copy(pdf)
        text = await rag.get_full_text(path)
        chunks = rag.chunk_text(text)
        if rag.index is None:
            rag.init_index(chunks)
        else:
            rag.extend_index(chunks)
    await cl.Message(content="Initial PDF index ready. You can ask questions or type 'add document' to upload more.").send()


@cl.on_message
async def main(message: cl.Message):
    content = message.content.strip().lower()
    if content in ("add document", "upload document"):
        files = await cl.AskFileMessage(
            content="Upload a PDF to add to the index:",
            accept=["application/pdf"],
            max_size_mb=20
        ).send()
        file = files[0]
        path = save_temp_copy(file.path)
        await cl.Message(content=f"Processing and indexing `{file.name}`...").send()
        text = await rag.get_full_text(path)
        chunks = rag.chunk_text(text)
        rag.extend_index(chunks)
        await cl.Message(content="Document added. Index updated!").send()
        return

    # Regular Q&A
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    # Generate on-the-fly retriever

    class Retriever:
        async def get_relevant_documents(self, q: str):
            snippets = rag.query(q, top_k=5)
            return [Document(page_content=s) for s in snippets]
    retriever = Retriever()
    # Build chain if not exist
    if chain is None:
        history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=history)
        chain = ConversationalRetrievalChain.from_llm(
            ChatAnthropic(model="claude-3-5-sonnet-20240620"),
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
        )
        cl.user_session.set("chain", chain)
    res = await chain.acall(message.content, callbacks=[cb])
    ans = res.get("answer", "")
    docs = res.get("source_documents", [])
    elements = [cl.Text(content=d.page_content,
                        name=f"src{i}", display="side") for i, d in enumerate(docs)]
    if elements:
        ans += "\nSources: " + ", ".join(e.name for e in elements)
    await cl.Message(content=ans, elements=elements).send()

