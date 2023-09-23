from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from rich import print

from ml_config import *


DB_FAISS_PATH = 'vectorstore/db_faiss'

embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small",
                                    model_kwargs={'device': 'cuda'})
db = FAISS.load_local(DB_FAISS_PATH, embeddings)


q = "I don't even remember when or why but I just feel empty. Can I fix this?"

results = similarity_search_results(q)
print(q)

def similarity_search_results(q, sep = '\n'):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                       model_kwargs=EMBEDDING_MODEL_ARGS)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    relevant_documents = db.similarity_search_with_relevance_scores(query=q, **SIMILARITY_SEARCH_KWARGS)

    context = []
    for document, score in relevant_documents:
        if score >= SIMILARITY_SEARCH_THRESHOLD:
            context.append(document.page_content)

    return sep.join(context)

