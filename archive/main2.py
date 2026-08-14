# Combined RAG Chatbot with Chainlit Interface
# This combines your async PDF RAG system with a chat interface

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
import pdfplumber
import chainlit as cl
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


CLAUDE_KEY = "REDACTED-ANTHROPIC-KEY"
os.environ['ANTHROPIC_API_KEY'] = CLAUDE_KEY

# Configuration for async processing
MAX_CONCURRENT_PDFS = 1
MAX_CONCURRENT_CHUNKS = 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CACHE_FOLDER = "txt_cache"
INDEX_FILE_PATH = 'index.faiss'
METADATA_FILE_PATH = 'meta.pkl'

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
        """Query the index asynchronously using sentence transformer embedding model."""
        def _embed_query():
            return self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=self.embedding_config.normalize_embeddings
            )[0]

        def _search_index(q_emb):
            distances, indices = index.search(np.array([q_emb], dtype="float32"), top_k)
            return [metadata[i] for i in indices[0]], distances[0]

        loop = asyncio.get_event_loop()
        q_emb = await loop.run_in_executor(self.executor, _embed_query)
        results, distances = await loop.run_in_executor(self.executor, _search_index, q_emb)

        return results, distances

    def chunk_text(self, full_text: str) -> List[str]:
        """Split text into semantically coherent chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        return splitter.split_text(full_text)

    def __del__(self):
        """Cleanup executor on destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

class RAGChatbot:
    """Chatbot that combines RAG retrieval with Claude for answering questions."""

    def __init__(self, rag_system: AsyncPDFRAG, index: faiss.Index, metadata: List[str]):
        self.rag_system = rag_system
        self.index = index
        self.metadata = metadata
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)

    async def get_response(self, question: str, chat_history: str = "") -> Tuple[str, List[str]]:
        """Get response from RAG system with Claude."""
        try:
            # Retrieve relevant chunks
            relevant_chunks, distances = await self.rag_system.query_index(
                question, self.index, self.metadata, top_k=5
            )

            # Create context from relevant chunks
            context = "\n\n".join(relevant_chunks)

            # Create prompt for Claude
            prompt = f"""You are a helpful assistant that answers questions based on the provided context from PDF documents.

Context from documents:
{context}

Chat History:
{chat_history}

Question: {question}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so and provide what information you can based on the available context.

Answer:"""

            # Get response from Claude
            response = await self.llm.ainvoke(prompt)

            return response.content, relevant_chunks

        except Exception as e:
            logger.error(f"Error getting response: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}", []

# --- Chainlit Interface ---

welcome_message = """🤖 Welcome to the Advanced PDF RAG Chatbot!

This chatbot can answer questions about your indexed PDF documents using:
- **FAISS** vector database for fast similarity search
- **Sentence Transformers** for embeddings
- **Claude** for intelligent responses

The system will automatically load your pre-built index and start answering questions!

**How to use:**
1. Simply type your question about the documents
2. The system will find relevant information from your PDFs
3. Claude will provide a comprehensive answer based on the context

**Example questions:**
- "What are the main topics covered in the documents?"
- "Can you summarize the key findings?"
- "What information is available about [specific topic]?"

Ready to chat! 🚀
"""

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chatbot when a new chat session starts."""

    # Show welcome message
    await cl.Message(content="🔄 **Initializing RAG Chatbot...**\n\nLoading embedding model and FAISS index...").send()

    try:
        # Initialize RAG system
        parsing_config = PDFParsingConfig()
        embedding_config = EmbeddingConfig(model_name=EMBEDDING_MODEL)

        rag_system = AsyncPDFRAG(
            cache_folder=CACHE_FOLDER,
            parsing_config=parsing_config,
            embedding_config=embedding_config
        )

        # Check if index exists
        if not os.path.exists(INDEX_FILE_PATH) or not os.path.exists(METADATA_FILE_PATH):
            error_msg = f"""❌ **Index files not found!**

Please make sure you have:
1. `{INDEX_FILE_PATH}` - FAISS index file
2. `{METADATA_FILE_PATH}` - Metadata file

Run your PDF ingestion script first to create these files.

**Expected files:**
- Index: `{INDEX_FILE_PATH}`
- Metadata: `{METADATA_FILE_PATH}`
"""
            await cl.Message(content=error_msg).send()
            return

        # Load the index and metadata
        await cl.Message(content="📚 **Loading FAISS index and metadata...**").send()
        index, metadata = await rag_system.load_index(INDEX_FILE_PATH, METADATA_FILE_PATH)

        # Initialize chatbot
        chatbot = RAGChatbot(rag_system, index, metadata)

        # Store in session
        cl.user_session.set("chatbot", chatbot)
        cl.user_session.set("chat_history", "")

        # Initialize memory for conversation
        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )
        cl.user_session.set("memory", memory)

        # Success message
        success_msg = f"""✅ **RAG Chatbot Ready!**

📊 **System Info:**
- **Documents indexed:** {len(metadata)} chunks
- **Embedding model:** {EMBEDDING_MODEL}
- **LLM:** Claude 3.5 Sonnet
- **Vector DB:** FAISS

{welcome_message}
"""

        await cl.Message(content=success_msg).send()

    except Exception as e:
        error_msg = f"""❌ **Initialization Error:**

{str(e)}

Please check:
1. Your index files exist and are valid
2. Your API key is set correctly
3. All dependencies are installed

**Required files:**
- `{INDEX_FILE_PATH}`
- `{METADATA_FILE_PATH}`
"""
        await cl.Message(content=error_msg).send()
        logger.error(f"Initialization error: {e}")

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""

    # Get chatbot from session
    chatbot = cl.user_session.get("chatbot")
    if not chatbot:
        await cl.Message(content="❌ **Error:** Chatbot not initialized. Please refresh the page.").send()
        return

    # Get chat history
    chat_history = cl.user_session.get("chat_history", "")

    # Show thinking message
    thinking_msg = cl.Message(content="🤔 **Thinking...**\n\nSearching through documents and generating response...")
    await thinking_msg.send()

    try:
        # Get response from RAG chatbot
        response, relevant_chunks = await chatbot.get_response(message.content, chat_history)

        # Update chat history
        new_history = f"{chat_history}\nHuman: {message.content}\nAssistant: {response}\n"
        cl.user_session.set("chat_history", new_history[-4000:])  # Keep last 4000 chars

        # Create text elements for source chunks
        text_elements = []
        if relevant_chunks:
            for idx, chunk in enumerate(relevant_chunks[:3]):  # Show top 3 chunks
                source_name = f"source_{idx+1}"
                # Truncate long chunks for display
                display_chunk = chunk[:500] + "..." if len(chunk) > 500 else chunk
                text_elements.append(
                    cl.Text(content=display_chunk, name=source_name, display="side")
                )

        # Add source information to response
        if text_elements:
            source_names = [text_el.name for text_el in text_elements]
            response += f"\n\n📚 **Sources:** {', '.join(source_names)}"
        else:
            response += "\n\n📚 **Sources:** No specific sources found"

        # Update the thinking message with the final response
        thinking_msg.content = response
        thinking_msg.elements = text_elements
        await thinking_msg.update()

    except Exception as e:
        error_response = f"❌ **Error processing your question:**\n\n{str(e)}\n\nPlease try rephrasing your question or check if the system is properly initialized."
        thinking_msg.content = error_response
        await thinking_msg.update()
        logger.error(f"Error in main: {e}")

# --- Health Check Endpoint ---
@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat ends."""
    logger.info("Chat session ended")

# --- Additional Features ---

@cl.action_callback("refresh_index")
async def refresh_index():
    """Action to refresh the index if needed."""
    await cl.Message(content="🔄 **Refreshing index...** (This feature can be implemented to reload the index)").send()

# --- Run Instructions ---
if __name__ == "__main__":
    # Make sure you have your index files ready before running this
    print("🚀 Starting RAG Chatbot with Chainlit...")
    print(f"📁 Expected index file: {INDEX_FILE_PATH}")
    print(f"📁 Expected metadata file: {METADATA_FILE_PATH}")
    print("\n💡 Make sure to:")
    print("1. Set your ANTHROPIC_API_KEY in the code")
    print("2. Have your FAISS index and metadata files ready")
    print("3. Install all required dependencies")
    print("\n🔧 To run: chainlit run this_file.py")
