from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import  InMemoryVectorStore, FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

pdf_directory = "pdfs/"


embeddings = OllamaEmbeddings(model="deepseek-r1:7.6b")

model = OllamaLLM(
    model="deepseek-r1:7.6b",
    temperature=0)

template = """
You are an assistant that answers questions. Using the following retrieved information, answer the user
question. If  you don't know the answer, say that you don't know, don't try to make up an answer.
Use up to three sentences, keeping the answer brief and concise.
Question: {question}
Context: {context}
Answer:
"""


def upload_pdf(file):
    """
    Uploads a pdf file to the pdf_directory.

    Args:
        file: the file to upload

    Returns:
        None
    """
    with open(pdf_directory + file.name, "wb") as f:
        f.write(file.getbuffer())



def create_vector_store(file_path):
    
    """
    Creates a vector store from a given pdf file.

    Args:
        file_path: the file path to the pdf file

    Returns:
        the vector store
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        add_starting_index=True
          )
    
    chunked_docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(chunked_docs, embeddings)
    return db


def retriev_docs(db,query,k=4):
    print(db.similarity_search(query))
    return db.similarity_search(query,k)


def question_pdf(question,documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})