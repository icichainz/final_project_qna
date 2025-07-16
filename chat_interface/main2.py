import os
import shutil
import tempfile
from typing import List

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
import chainlit as cl
from PyPDF2 import PdfReader
import docxpy

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)


CLAUDE_KEY= "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"

os.environ['OPENAI_API_KEY'] = CLAUDE_KEY

welcome_message = """Welcome to the PDF Question and Answer.
The app allows you to add a PDF, text file, or docx and chat over it!
To get started:
1. Upload a PDF, text file, or docx.
2. Wait for the file to be processed.
3. Ask a question about the file.
4. Enjoy yourself!
"""

def save_temp_copy(uploaded_file_path):
    """Save file to temporary dir."""
    tempdir = tempfile.mkdtemp()
    temp_file_path = shutil.copy(uploaded_file_path, tempdir)
    print(f"The temporary file path is: {temp_file_path}")
    return temp_file_path

def process_word(path_to_file):
    """Process the word document."""
    return docxpy.process(path_to_file)

def process_pdf(path_to_file):
    """Process the pdf document."""
    text = ""
    reader = PdfReader(path_to_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text

@cl.on_chat_start
async def on_chat_start():
    elements = [
        cl.Image(name="image1", display="inline", path="./robot.jpeg")
    ]
    await cl.Message(
        content="Hello there, Welcome to AskAnyQuery related to Data!",
        elements=elements
    ).send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=[
                "text/plain",
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Decode the file
    if file.name.endswith('.txt'):
        with open(file.path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file.name.endswith('.pdf'):
        pdf_path = save_temp_copy(file.path)
        text = process_pdf(pdf_path)
    elif file.name.endswith('.docx'):
        docx_path = save_temp_copy(file.path)
        text = process_word(docx_path)
    else:
        text = ""

    # Split the text into chunks
    texts = text_splitter.split_text(text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    embeddings = HuggingFaceEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatAnthropic(model="claude-3-5-sonnet-20240620"),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name, display="side")
            )
        source_names = [text_el.name for text_el in text_elements]
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()


