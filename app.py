from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Updated imports for document loaders and embeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma  # Ensure you've installed via: pip install -U langchain-chroma

# Import other LangChain components
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# --- Helper Functions ---
# 
def initialize_llm():
    llm = ChatGroq(
        temperature=0.1,
        groq_api_key="YOUR_GROQ_API_KEY",
        model_name="llama-3.3-70b-versatile"
    )
    return llm


def create_vector_db():
    loader = DirectoryLoader("./data/", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    vector_db.persist()
    print("ChromaDB created and data saved")
    return vector_db

# Initialize LLM and vector DB.
llm = initialize_llm()
db_path = "./chroma_db"
if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

# Setup the conversational retrieval chain.
def setup_conversational_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    # Define the prompt template including the chat history.
    prompt_templates = (
        "You are a compassionate, empathetic, and supportive mental health counselor. "
        "Your role is to listen attentively and provide general coping strategies. "
        "Consider the previous conversation:\n\n"
        "{chat_history}\n\n"
        "Here is some context to help answer the user's question:\n"
        "{context}\n\n"
        "User: {question}\n"
        "Chatbot:"
    )
    PROMPT = PromptTemplate(
        template=prompt_templates,
        input_variables=["chat_history", "context", "question"]
    )
    
    # The retriever automatically provides the "context" from the vector DB.
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return conversation_chain

conversation_chain = setup_conversational_chain(vector_db, llm)

# --- FastAPI App Initialization ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production as needed.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from langchain.schema import SystemMessage

class ChatRequest(BaseModel):
    chat_history: str  # Summarized conversation history as a string
    question: str

@app.post("/chat")
def chat(request: ChatRequest):
    # Convert the summary string into a list of messages.
    # If no summary is provided, use an empty list.
    chat_history_list = [SystemMessage(content=request.chat_history)] if request.chat_history else []
    
    result = conversation_chain({
        "chat_history": chat_history_list,
        "question": request.question
    })
    return {"response": result["answer"]}


class SummarizeRequest(BaseModel):
    text: str

summary_template = (
    "Please summarize the following text into a concise summary "
    "that does not exceed 300 words, while retaining all the essential details and context:\n\n"
    "{chat_history}"
)

summary_prompt = PromptTemplate(
    input_variables=["chat_history"],
    template=summary_template
)

from langchain.schema import HumanMessage

@app.post("/summarize")
def summarize_text(req: SummarizeRequest):
    formatted_prompt = summary_prompt.format(chat_history=req.text)
    # Wrap the prompt in a HumanMessage and invoke the model.
    summary_text = llm.invoke([HumanMessage(content=formatted_prompt)])
    return {"summary": summary_text}
