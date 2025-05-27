# RAG Index Pipeline with LangChain and FAISS

This project creates a data ingestion pipeline (Index pipeline) for RAG (Retrieval-Augmented Generation) using [LangChain](https://www.langchain.com/) components. It loads the document, and splits them into manageable chunks, generates embeddings, and stores them in a FAISS vector database.

## 📦 Features

- Splits text into overlapping chunks for better semantic embedding
- Embeddings using `GPT4AllEmbeddings` (can be replaced with any other embeddings from [LangChain Embeddings](https://python.langchain.com/docs/integrations/text_embedding/))
- Stores the vectors in a FAISS index for fast similarity search
- Supported documents include raw text file (`.txt`) and PDF document (`.pdf`) formats


## 🚀 Technologies

- **LangChain** 
- **FAISS** 

## 📦 Steps

1. **Create a Virtual Environment**
    
    ```bash 
    virtualenv ~/IndexPipeline
    source ~/IndexPipeline/bin/activate

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt

3. **Replace file name here**

   ```bash
    file = 'file.pdf'

3. **Run**
    ```bash
    python index_pipeline.py
