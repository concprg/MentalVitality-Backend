from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple, Union
from os.path import sep as PathSep

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import LLMChain, RetrievalQA, QAWithSourcesChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory


from pydantic import BaseModel

DATA_PATH = 'data/'
DB_FAISS_PATH = '/chatbot/vectorstore/db_faiss'

MODEL_PATH = "models/mythologic-mini-7b.ggmlv3.q3_K_L.bin"
MODEL_NAME = MODEL_PATH.partition(".")[0].partition(PathSep)[-1]
MODEL_KWARGS = {"max_new_tokens": 1024, "temperature": 0.8}

EMBEDDING_MODEL = "thenlper/gte-small"
EMBEDDING_MODEL_ARGS = model_kwargs={'device': 'cuda'}

SIMILARITY_SEARCH_KWARGS = {'k': 3}

therapist_prompt_instruction = """You are a responsive and skilled therapist taking care of a patient who is looking for guidance and advice on managing their emotions, stress, anxiety and other mental health issues through chat based therapy. Attentively listen to the patient and answer the patient's questions in an empathetic and non-judgemental tone, and do not judge the patient for any issues they are facing. Offer acceptance, support, and care for the patient, regardless of their circumstances or struggles.  Make them comfortable and ask open ended questions in an empathetic manner that encourages self reflection. Try to further understand the patient's problem and to help them solve their problems if and only if they want you to solve it. Also try to avoid giving false or misleading information, and caveat when you entirely sure about the right answer.
Additionally, use the following context to chat with the user:
{chat_history}
Respond to the patient and give them a concise response, asking open questions that encourage reflection, self-introspection and deep thought.
"""

therapist_prompt_input = "{question}"


instruct_prompt_template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input: 
{input}

### Response:

'''


def set_custom_prompt(custom_prompt_template):
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['question', "chat_history"])
    return prompt

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = MODEL_PATH,
        model_type="llama",
        max_new_tokens = 4096,
        temperature = 0.75,
        gpu_layers = 16
    )
    return llm


chat_history = []

def conv_retr_chain(llm, prompt, db, lexical_retriever=None):
    # retriever = db.as_retriever(search_kwargs={'k': 3})
    # if lexical_retriever != None:
    #     retriever = EnsembleRetriever(retrievers = [db.as_retriever(search_kwargs = {"k" :2 }), lexical_retriever])
    conv_retr_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                        retriever = db.as_retriever(search_kwargs={'k': 3}),
                                        return_source_documents = True,
                                        )
    return conv_retr_chain


#QA Model Function

def therapist_bot_with_rag():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                       model_kwargs=EMBEDDING_MODEL_ARGS)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    llm = load_llm()
    qa_prompt = set_custom_prompt(custom_prompt_template=instruct_prompt_template.format(**{"instruction":therapist_prompt_instruction, "input":therapist_prompt_input}))
    qa = conv_retr_chain(llm, qa_prompt, db)

    return qa

def generate_therapist_response_with_rag(query):
    qa_result = therapist_bot_with_rag()
    while True:
        if query == "exit" or query == "quit" or query == "q" or query == "f":
            print('Exiting')
        if query == '':
            continue
        result = qa_result(
            {"question": query, "chat_history": chat_history})
        response = result["answer"]
        chat_history.append((query, result["answer"]))
        return response

def generate_instruct_response(prompt_instruction, prompt_input):
    llm = load_llm()
    base_chain = LLMChain(llm = llm, prompt = set_custom_prompt(instruct_prompt_template.format(**{"instruction":prompt_instruction, "input":prompt_input})))
    result = base_chain.run()
    return result



def generate_openai_response(message):
    choices = []
    
    choices.append({
            "role":"assistant",
            "content":message,
        })
    data = {choices}
    return data
