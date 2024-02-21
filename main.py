import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import SingleStoreDB
from langchain_openai import OpenAIEmbeddings
import google.generativeai as gemini
from lighthouz import Lighthouz
from lighthouz.benchmark import Benchmark
from lighthouz.app import App
from lighthouz.evaluation import Evaluation

# Load the PDF document
loader = PyPDFLoader("superbowl.pdf")
data = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# Initialize embeddings
embeddings = OpenAIEmbeddings()


os.environ["OPENAI_API_KEY"] = "INSERT_KEY"  # Enter your OpenAI key to be used in the RAG model.
os.environ["GEMINI_API_KEY"] = "INSERT_KEY"
gemini.configure(api_key=os.environ["GEMINI_API_KEY"])
os.environ["SINGLESTOREDB_URL"] = "SINGLESTOREDB_DATABASE_URL"  # Set up the SingleStoreDB connection
SingleStoreVectorDB = SingleStoreDB.from_documents(texts, embeddings, table_name="notebook")

# Initialize the Gemini model
model = gemini.GenerativeModel('gemini-pro')

# Initialize Lighthouz
LH = Lighthouz("LIGHTHOUZ-API-KEY")
benchmark_id = "65d54f4a1f281c52d90d51c3"  # add your own benchmark
app_id = '65d552dc9b50b4563f339eec'  # add your own app


# Modularized RAG Model function
def rag_model(query):
    # Find documents that correspond to the query
    docs = SingleStoreVectorDB.similarity_search(query)
    if docs:
        context = docs[0].page_content
    else:
        context = "No relevant information found."

    # Prepare the prompt for the Gemini model
    prompt_template = f"""
        You are a helpful Assistant who answers users' questions based on contexts given to you.

        Keep your answer short and to the point.

        The evidence is the context of the PDF extract with metadata.

        Reply "Not applicable" if the text is irrelevant.

        The user query: 
        {query}

        The document context:
        {context}
    """

    # Generate a response
    response = model.generate_content(prompt_template)

    return response.text


# Streamlit app
def chatbot():
    st.title("RAG Chatbot")
    user_input = st.text_input("Ask me anything about the Superbowl:")

    if user_input:
        response = rag_model(user_input)
        st.write(response)

    # Button for evaluating the RAG Chat Bot
    if st.button('Evaluate RAG Chat Bot'):
        # Initialize evaluation
        evaluation = Evaluation(LH)
        results = evaluation.evaluate_rag_model(
            response_function=rag_model,
            benchmark_id=benchmark_id,
            app_id=app_id,
        )


if __name__ == "__main__":
    chatbot()
