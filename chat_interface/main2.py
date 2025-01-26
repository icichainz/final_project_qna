import os
from pydoc import doc
import shutil
from tempfile import tempdir
import tempfile
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI

from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from  langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
import chainlit as cl
from PyPDF2 import PdfReader
import docxpy


""" 
In RecursiveCharacterTextSplitter, the chunk_overlap value specifies 
how many characters from the end of 
one chunk are included at the beginning of the next chunk.
"""
# The chuck size specify how many characters at the end of off one chunck if included at the begining of the next chunck.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

welcome_message = """Welcome to the  PDF Question and Answer. \n The app will alloww you to add a pdf or a text file and chat over ! To get started:
1. Upload a PDF ,text file or a docx. 
2. Wait for the file to be processed.
3. Ask a question about the file
4. Enjoy yourself!
"""

@cl.on_chat_start
async def on_chat_start():
    # sending the correct elements to the chat.
    
    elements = [
    cl.Image(name="image1", display="inline", path="./robot.jpeg")
    ]
    
    await cl.Message(content="Hello there, Welcome to AskAnyQuery related to Data!",
                     elements=elements).send()
    
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain",
                    "application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()
    
    
    # Decode the file if its .txt file
    if file.name.endswith('.txt'):
        with open(file.path, "r", encoding="utf-8") as f:
            text = f.read()
            
    def save_temp_copy(uplaoded_file_path):
        """ Save file to temporary dir. """
        tempdir = tempfile.mkdtemp()
        temp_file_path = shutil.copy(uplaoded_file_path,tempdir)
        print(f"The temporary file path is: {temp_file_path}")
        return temp_file_path

    def process_word(path_to_file):
        """ Process the word document."""
        text =  docxpy.process(path_to_file)
        return text  

    def process_pdf(path_to_file):
        """ Process the pdf document."""
        text= " "
        reader = PdfReader(path_to_file)
        for page in reader.pages:
            text += page.extract_text() +"\n\n"
        return text  

    # Decode pdf file.
    if file.name.endswith('.pdf'):
        pdf_path = save_temp_copy(file.path)
        text = process_pdf(pdf_path)
    
    # Decode docs file.
    if file.name.endswith('.docx'):
        docx_path = save_temp_copy(file.path)
        text = process_word(docx_path)
        
    # Split the text into chunks
    # The text is split and using recursive text splitter . 
    texts = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = HuggingFaceEmbeddings()
    # The docsearch use chroma to 
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

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        #ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        ChatAnthropic(model="claude-3-5-sonnet-20240620",),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
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
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name, display="side")
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
