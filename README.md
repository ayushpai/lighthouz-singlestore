# RAG Chatbot + Evaluation w/ Lighthouz & SingleStore

Retrieval Augmented Generation (RAG) is one of the most popular ways to increase the accuracy of Large Language Models (LLMs) and reduce hallucinations. However, even with RAG-based systems, LLMs are prone to many issues. Understanding these issues with standardized benchmarks is important in order to improve a model or the documents in RAG. LightHouz AI allows you to evaluate your LLM across 6 benchmark categories: Hallucination Tests, Out of Context, Prompt Injection, PII Leak, Toxicity, and Bias. Lighthouz AutoBench automatically generates benchmarks to evaluate your RAG Application based on the documents you upload. It also facilitates AutoEvals of those benchmarks comparing the expected result of a query to the actual response. You can also compare multiple LLMs on the same benchmark to see which performs better.

This demo allows you to run a RAG Chatbot in a Streamlit interface and evaluate the chatbot using LightHouz AI.

## Tech Stack
- LangChain -  Pre-processes and formats text data, making it suitable for embedding generation
- OpenAI Embeddings - Generates vectorized forms of the documents.
- SingleStoreDB - Stores the prepared embeddings in a vector database.
- LLM of Your Choice (GPT-4, Gemini Pro) - Takes in user prompt + context from retrieval and generates output.
- LighthouzAI - Evaluates RAG chatbot responses.

## Setup
1. Install the `requrements.txt`
2. Set up environment variables for Google Gemini and OpenAI API key. (`export OPENAI_API_KEY='your-api-key-here'`).
3. Add your SingleStoreDB database URL to line 26 to establish the database connection.
4. Replace the PDF document on line 15 with any document of your choice for RAG (or keep this one to test it out!)
5. Add your LightHouz API Key on line 31 of `main.py`: `LH = Lighthouz("LH-XRjjxBxtYjXPQqwpPJ0WyHcc0tjBx6vy")`.
6. Generate a new benchmark for your RAG app on the LightHouz AutoBench Dashboard. Enter the `benchmark_id` on line 32.
7. Create new apps in the LightHouz Dashboard for `gpt-4` and `gemini-pro`. Enter the `app_id`s onto lines 33-34.
8. That's it! You're ready to use your chatbot and evaluations!

## Execution
`streamlit run main.py`

## Resources
- Slides From this Demo: https://docs.google.com/presentation/d/1JG57ZVd0_zKhzM6SkKtes72SrMsKh_wCd_O6vCfuDtw/edit?usp=sharing
- LightHouz Documentation: https://lighthouz.ai/docs/

