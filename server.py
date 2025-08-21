# main.py
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# LangChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def setup_environment():
    load_dotenv()

# ------------------------------
# Document Loading
# ------------------------------
def load_documents(file_path, file_type):
    documents = []
    try:
        if file_type == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        elif file_type == "pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split()
        elif file_type == "csv":
            loader = CSVLoader(file_path=file_path, encoding="utf-8")
            documents = loader.load()
        else:
            print(f"Unsupported file type: {file_type}")
            return []
        print(f"Loaded {len(documents)} document(s) from {file_path}.")
    except Exception as e:
        print(f"Error loading {file_type} file '{file_path}': {e}")
    return documents

# ------------------------------
# Create RAG Pipeline with MongoDB Atlas
# ------------------------------
def create_rag_pipeline(docs):
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    print(f"Split into {len(texts)} chunks.")

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    mongo_client = MongoClient(os.getenv("MONGODB_ATLAS_URI"))
    db_name = "ragdb"
    collection_name = "rag_chunks"
    collection = mongo_client[db_name][collection_name]

    # Only embed if not already present
    if collection.estimated_document_count() == 0:
        print("Generating and storing embeddings in MongoDB...")
        vectorstore = MongoDBAtlasVectorSearch.from_documents(
            documents=texts,
            embedding=gemini_embeddings,
            collection=collection,
            index_name="vector_index"
        )
    else:
        print("Using existing embeddings from MongoDB...")
        vectorstore = MongoDBAtlasVectorSearch(
            embedding=gemini_embeddings,
            collection=collection,
            index_name="vector_index"
        )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )
    print("Retriever created with k=3.",retriever)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the following question based ONLY on the provided context.
    If the answer is not found in the context, say so.

    <context>
    {context}
    </context>

    Question: {input}

    Answer:
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain
# Add this to server.py
def ask_chatbot(question, rag_chain):
    if not rag_chain:
        return {"answer": "RAG chain not initialized."}
    
    try:
        response = rag_chain.invoke({"input": question})
        return {"answer": response["answer"]}
    except Exception as e:
        print(f"Error during chatbot invocation: {e}")
        return {"answer": "Sorry, an error occurred while processing your question."}

# # ------------------------------
# # Chatbot Query
# # ------------------------------
# def ask_chatbot(query, chain):
#     if not chain:
#         return "RAG chain not initialized."
#     response = chain.invoke({"input": query})
#     return response.get("answer", "No answer found.")

# ------------------------------
# Main
# ------------------------------
# if __name__ == "__main__":
#     setup_environment()

#     if not os.getenv("GOOGLE_API_KEY") or not os.getenv("MONGODB_ATLAS_URI"):
#         print("Missing API keys. Set GOOGLE_API_KEY and MONGODB_ATLAS_URI in .env.")
#         exit()

#     DATA_FILE_PATH = "RAG_based_Chatbot-main/data/dataset_childrenbook.pdf"
#     DATA_FILE_TYPE = "pdf"

#     documents = load_documents(DATA_FILE_PATH, DATA_FILE_TYPE)
#     if not documents:
#         print("No documents loaded.")
#         exit()

#     rag_chain = create_rag_pipeline(documents)
#     if not rag_chain:
#         print("Failed to create RAG pipeline.")
#         exit()

#     print("\n✅ Chatbot ready!")
#     sample_questions = [
#         "What is the Goldilocks Principle mentioned in the paper?",
#         "What is the Children's Book Test (CBT)?"
#     ]

#     for q in sample_questions:
#         print(f"\nQ: {q}")
#         print(f"A: {ask_chatbot(q, rag_chain)}")
#     print("\nYou can now ask your own questions!")