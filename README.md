# RAG Chatbot + Evaluation w/ Lighthouz & SingleStore

Retrieval Augmented Generation (RAG) is one of the most popular ways to increase the accuracy of Large Language Models (LLMs) and reduce hallucinations. However, even with RAG-based systems, LLMs are prone to many issues. Understanding these issues with standardized benchmarks is important in order to improve a model or the documents in RAG. LightHouz AI allows you to evaluate your LLM across 6 benchmark categories: Hallucination Tests, Out of Context, Prompt Injection, PII Leak, Toxicity, and Bias. Lighthouz AutoBench automatically generates benchmarks to evaluate your RAG Application based on the documents you upload. It also facilitates AutoEvals of those benchmarks comparing the expected result of a query to the actual response. You can also compare multiple LLMs on the same benchmark to see which performs better.

This demo allows you to run a RAG Chatbot in a Streamlit interface and evaluate the chatbot using LightHouz AI.

## Tech Stack
- LangChain -  Pre-processes and formats text data, making it suitable for embedding generation
- OpenAI Embeddings - Generates vectorized forms of the documents.
- SingleStoreDB - Stores the prepared embeddings in a vector database.
- LLM of Your Choice (GPT-4, Gemini Pro) - Takes in user prompt + context from retrieval and generates output.
- LighthouzAI - Evaluates RAG chatbot responses.
