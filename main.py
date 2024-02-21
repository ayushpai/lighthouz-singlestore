import os
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import SingleStoreDB
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from lighthouz import Lighthouz
from lighthouz.evaluation import Evaluation

# Load the PDF document
loader = PyPDFLoader("superbowl.pdf")
data = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# Generate embeddings
embeddings = OpenAIEmbeddings()

# Set up the SingleStoreDB connection and vector database
os.environ["SINGLESTOREDB_URL"] = "SINGLESTOREDB_URL"  # insert SingleStoreDB Database URL
SingleStoreVectorDB = SingleStoreDB.from_documents(texts, embeddings, table_name="notebook")

# Initialize Lighthouz
LH = Lighthouz("LIGHTHOUZ_API_KEY")  # insert LightHouz API key
benchmark_id = 'insert_benchmark_id'
gemini_app_id = 'insert_gemini_app_id'  # gemini app
gpt4_app_id = 'insert_gpt4_app_id'  # gpt4 app


# Modularized RAG Model function
def rag_model(query, model):
    # Find documents that correspond to the query
    docs = SingleStoreVectorDB.similarity_search(query)
    if docs:
        context = docs[0].page_content
    else:
        context = "No relevant information found."

    # Prepare the prompt for the LLM
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

    # Generate a response based on the model
    if model == 'gemini-pro':
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
        return llm.invoke(prompt_template).content
    elif model == 'gpt-4':
        llm = OpenAI()
        return llm.invoke(prompt_template)


def gemini_response_function(query: str) -> str:
    return rag_model(query, 'gemini-pro')


def gpt4_response_function(query: str) -> str:
    return rag_model(query, 'gpt-4')


# Streamlit app
def chatbot():
    st.title("RAG Chatbot")

    model = st.radio(
        "Choose a model to generate responses:",
        ('gemini-pro', 'gpt-4')
    )

    user_input = st.text_input("Ask me anything about the Superbowl:")

    if user_input:
        response = rag_model(user_input, model)
        st.write(response)

    # Button for evaluating the RAG Chat Bot
    if st.button('Evaluate ' + model + ' RAG Chat Bot'):
        # Initialize evaluation
        evaluation = Evaluation(LH)

        response_function = None
        app_id = None
        if model == 'gemini-pro':
            response_function = gemini_response_function
            app_id = gemini_app_id
        elif model == 'gpt-4':
            response_function = gpt4_response_function
            app_id = gpt4_app_id

        results = evaluation.evaluate_rag_model(
            response_function=response_function,
            benchmark_id=benchmark_id,
            app_id=app_id,
        )
    elif st.button('Compare GPT-4 and Gemini Evaluations'):
        evaluation = Evaluation(LH)
        results = evaluation.evaluate_multiple_rag_models(
            response_functions=[gemini_response_function, gpt4_response_function],
            benchmark_id=benchmark_id,
            app_ids=[gemini_app_id, gpt4_app_id],
        )


if __name__ == "__main__":
    chatbot()
