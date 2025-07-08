import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Configuration
MODEL = "gpt-4o-mini"
DB_NAME = "vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 10
LLM_TEMPERATURE = 0.2

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")
os.environ['OPENAI_API_KEY'] = api_key

def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc

text_loader_kwargs = {'encoding': 'utf-8'}

def load_and_process_documents():
    folders = glob.glob("knowledge-base/*")
    
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])
    
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Total number of chunks: {len(chunks)}")
    print(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}")
    
    return chunks

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
    
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")
    
    return vectorstore

def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=LLM_TEMPERATURE, model_name=MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    custom_prompt_template = """You are an expert assistant helping employees of Insurellm, an insurance technology company. You must always answer based **only** on the provided context. Do not make up answers. If the context is unclear,
do your best to provide a helpful and accurate response grounded in what you know from the documents. Avoid saying "I don't know" unless absolutely nothing in the context is relevant. Try to interpret the question charitably
and helpfully, while staying factual. Chat History: {chat_history} Context: {context} Question: {question} Helpful Answer:"""
 
    custom_prompt = PromptTemplate(
        input_variables=["question", "context", "chat_history"],
        template=custom_prompt_template
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

def main():
    chunks = load_and_process_documents()
    vectorstore = create_vectorstore(chunks)
    conversation_chain = create_conversation_chain(vectorstore)
    
    def chat(question, history):
        result = conversation_chain.invoke({"question": question})
        return result["answer"]
    
    gr.ChatInterface(chat, type="messages").launch(inbrowser=True)

if __name__ == "__main__":
    main()