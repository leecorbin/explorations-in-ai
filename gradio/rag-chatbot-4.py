#!/usr/bin/env python
# coding: utf-8

# ## Expert Knowledge Worker
# 
# ### A question answering agent that is an expert knowledge worker
# ### To be used by employees of Insurellm, an Insurance Tech company
# ### The agent needs to be accurate and the solution should be low cost.
# 
# This project will use RAG (Retrieval Augmented Generation) to ensure our question/answering assistant has high accuracy.

# imports

import os
import glob
from dotenv import load_dotenv
import gradio as gr

# imports for langchain

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# price is a factor for our company, so we're going to use a low cost model

MODEL = "gpt-4o-mini"
db_name = "vector_db"

# Load environment variables in a file called .env

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# Read in documents using LangChain's loaders
# Take everything in all the sub-folders of our knowledgebase

folders = glob.glob("knowledge-base/*")

# With thanks to CG and Jon R, students on the course, for this fix needed for some users 
text_loader_kwargs = {'encoding': 'utf-8'}
# If that doesn't work, some Windows users might need to uncomment the next line instead
# text_loader_kwargs={'autodetect_encoding': True}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


len(chunks)

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")


# ## A sidenote on Embeddings, and "Auto-Encoding LLMs"
# 
# We will be mapping each chunk of text into a Vector that represents the meaning of the text, known as an embedding.
# 
# OpenAI offers a model to do this, which we will use by calling their API with some LangChain code.
# 
# This model is an example of an "Auto-Encoding LLM" which generates an output given a complete input.
# It's different to all the other LLMs we've discussed today, which are known as "Auto-Regressive LLMs", and generate future tokens based only on past context.
# 
# Another example of an Auto-Encoding LLMs is BERT from Google. In addition to embedding, Auto-encoding LLMs are often used for classification.
# 
# ### Sidenote
# 
# In week 8 we will return to RAG and vector embeddings, and we will use an open-source vector encoder so that the data never leaves our computer - that's an important consideration when building enterprise systems and the data needs to remain internal.

# Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
# Chroma is a popular open source Vector Database based on SQLLite

embeddings = OpenAIEmbeddings()

# If you would rather use the free Vector Embeddings from HuggingFace sentence-transformers
# Then replace embeddings = OpenAIEmbeddings()
# with:
# from langchain.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Delete if already exists

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Create vectorstore

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Get one vector and find how many dimensions it has

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")


# Prework

result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in doc_types]

# # Time to use LangChain to bring it all together

# <table style="margin: 0; text-align: left;">
#     <tr>
#         <td style="width: 150px; height: 150px; vertical-align: middle;">
#             <img src="../important.jpg" width="150" height="150" style="display: block;" />
#         </td>
#         <td>
#             <h2 style="color:#900;">PLEASE READ ME! Ignoring the Deprecation Warning</h2>
#             <span style="color:#900;">When you run the next cell, you will get a LangChainDeprecationWarning 
#             about the simple way we use LangChain memory. They ask us to migrate to their new approach for memory. 
#             I feel quite conflicted about this. The new approach involves moving to LangGraph and getting deep into their ecosystem.
#             There's a fair amount of learning and coding in LangGraph, frankly without much benefit in our case.<br/><br/>
#             I'm going to think about whether/how to incorporate it in the course, but for now please ignore the Depreciation Warning and
#             use the code as is; LangChain are not expected to remove ConversationBufferMemory any time soon.
#             </span>
#         </td>
#     </tr>
# </table>

# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


query = "Can you describe Insurellm in a few sentences"
result = conversation_chain.invoke({"question":query})
print(result["answer"])


# set up a new conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# ## Now we will bring this up in Gradio using the Chat interface -
# 
# A quick and easy way to prototype a chat with an LLM


# Wrapping in a function - note that history isn't used, as the memory is in the conversation_chain

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

# And in Gradio:

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)




