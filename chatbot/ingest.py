from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os

DATA_PATH = 'data/'
DB_FAISS_PATH = 'chatbot/vectorstore/db_faiss'
EMBEDDING_MODEL = "thenlper/gte-small"
EMBEDDING_MODEL_ARGS = model_kwargs={'device': 'cuda'}

# Create vector database
def create_vector_db(overwrite = True):
    if not overwrite and os.path.exists("chatbot/vectorstore/db_faiss/index.faiss"):
        return
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=40)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                       model_kwargs=EMBEDDING_MODEL_ARGS)

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()

