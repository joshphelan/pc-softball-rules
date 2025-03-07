# PC Softball Rules Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about softball rules based on the USSSA Softball Rulebook and Panama City Community rules.

## Features

- Interactive chat interface built with Streamlit
- RAG architecture using LangChain
- Prioritization of Panama City Community rules over USSSA rules
- Direct links to specific pages in the USSSA rulebook PDF
- Source citations with section paths for community rules
- Additional relevant page suggestions for further reading

## Architecture

This application uses a RAG (Retrieval-Augmented Generation) architecture:

1. **Data Ingestion**: The PDF rulebook and markdown community rules are processed and split into chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings using Sentence Transformers
3. **Vector Storage**: Embeddings are stored in a Chroma vector database
4. **Retrieval**: When a question is asked, the system performs a similarity search to retrieve relevant text chunks from the vector database
5. **Generation**: An LLM generates answers based on the retrieved context, prioritizing community rules

## Requirements

- Python 3.9+
- OpenAI API key

## Project Structure

```
pc-softball-rules/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
├── README.md               # This file
├── data/                   # Source documents
│   ├── usssa-slowpitch-2025-rulebook-final.pdf  # USSSA rulebook
│   └── panama_city_community_rules.md           # Community rules
├── pics/                   # Images
│   └── yelladawg.png       
├── src/
│   ├── utils/
│   │   ├── pdf_loader.py   # PDF loading utilities
│   │   ├── text_splitter.py # Text chunking utilities
│   │   └── embeddings.py   # Embedding model utilities
└── chroma_db/              # Vector database (created on first run)
