# finAId Codebase 


This codebase contains Python scripts for a financial analysis application called finAId. The application uses a language model API to generate responses based on user queries, and it also includes functionality for retrieval-augmented generation (RAG) and financial metric calculations.

## Base block

agent.py

- LLMClient: A class for making requests to a language model API.
- Agent: A class representing an individual agent that can send a request to the language model API based on a given prompt.
- AgentManager: A class for creating and managing multiple agents, running them, and getting a response from a RAG system

## Usage

pip install -U uv

uv sync

streamlit run router.py 