# SQLite version hack for Chroma
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import time

# Set page configuration
st.set_page_config(
    page_title="PC Softball Rules Assistant",
    page_icon="⚾",
    layout="wide"
)

# Hide GitHub icon and other Streamlit UI elements
hide_streamlit_style = """
<style>
/* Hide GitHub icon */
.stGitHubLink {
    visibility: hidden;
}

/* Hide hamburger menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.utils.pdf_loader import load_and_process_pdf
from src.utils.text_splitter import split_text, split_markdown
from src.utils.embeddings import get_embeddings


# Initialize session state for tracking costs
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0
if "current_cost" not in st.session_state:
    st.session_state.current_cost = 0
if "last_query_cost" not in st.session_state:
    st.session_state.last_query_cost = 0

# Load yelladawg image and convert to base64 for inline display
def get_image_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            import base64
            return base64.b64encode(img_file.read()).decode()
    return ""

# Get base64 encoded image
yelladawg_base64 = get_image_base64("pics/yelladawg.png")

# Create title with inline image
if yelladawg_base64:
    title_html = f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <h1> <img src="data:image/png;base64,{yelladawg_base64}" style="height: 1.2em; vertical-align: middle;">
        PC Softball Rules Assistant
        </h1>
    </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)
else:
    st.title("⚾ PC Softball Rules Assistant")

# Description
st.markdown("Find information about the rules of the Panama City softball league, based on USSSA and local community rules. For best results, ask specific questions about rules, equipment, field dimensions, or game procedures.")

# Add sidebar with information FIRST
with st.sidebar:
    st.header("About")
    
    # Usage statistics
    st.subheader("Usage Stats")
    st.metric("Current Search Cost", f"${st.session_state.last_query_cost:.6f}")
    st.metric("Total Session Cost", f"${st.session_state.total_cost:.6f}")
    
    # RAG explanation
    st.subheader("How It Works")
    st.markdown("""
    1. **Retrieval**: LangChain finds relevant rules
    2. **Generation**: GPT-4o mini creates answers
    """)
    
    # Add USSSA rulebook link
    st.subheader("Resources")
    st.markdown("[USSSA Slowpitch Rulebook 2025](http://cms.usssa.net/wp-content/uploads/sites/2/2025/01/usssa-slowpitch-2025-rulebook-final.pdf)")
    
    # Add developer credit at the bottom
    st.markdown("---")
    st.markdown("*Developed by [Josh Phelan](https://github.com/joshphelan/)*")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to load documents and create vector store
@st.cache_resource
def load_vector_store():
    # Always recreate the vector store to include both USSSA and Panama City Community rules
    if os.path.exists("./chroma_db"):
        # Load existing vector store
        embeddings = get_embeddings()
        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        return vector_store
    else:
        # Load USSSA rulebook
        usssa_pdf_path = "data/usssa-slowpitch-2025-rulebook-final.pdf"
        usssa_documents = load_and_process_pdf(usssa_pdf_path)
        
        # Add source metadata
        for doc in usssa_documents:
            doc.metadata["source"] = "USSSA"
            doc.metadata["priority"] = "low"
        
        # Load Panama City Community rules if available
        community_docs = []
        if os.path.exists("data/panama_city_community_rules.md"):
            with open("data/panama_city_community_rules.md", "r") as f:
                community_text = f.read()
            
            # Split the markdown text based on headers
            community_docs = split_markdown(community_text)
            
            # Update metadata for all community docs
            for doc in community_docs:
                doc.metadata["source"] = "Panama City Community"
                doc.metadata["priority"] = "high"
                
                # Add section information to metadata for better retrieval
                section_info = []
                if "section" in doc.metadata:
                    section_info.append(doc.metadata["section"])
                if "subsection" in doc.metadata:
                    section_info.append(doc.metadata["subsection"])
                if "subsubsection" in doc.metadata:
                    section_info.append(doc.metadata["subsubsection"])
                
                # Add section path to metadata
                if section_info:
                    doc.metadata["section_path"] = " > ".join(section_info)
        
        # Combine documents
        all_docs = community_docs + usssa_documents
        
        # Split text into chunks
        chunks = split_text(all_docs)
        
        # Create embeddings and vector store
        embeddings = get_embeddings()
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        # No need to call persist() as docs are automatically persisted
        return vector_store

# Function to format documents for display in the context
def format_docs(docs):
    # Sort documents by priority (community rules first)
    docs.sort(key=lambda d: d.metadata.get("priority", "low"), reverse=True)
    
    # Format each document with its source
    formatted_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        section_info = ""
        if source == "Panama City Community" and "section_path" in doc.metadata:
            section_info = f" ({doc.metadata['section_path']})"
        elif source == "USSSA" and "page" in doc.metadata:
            section_info = f" (Page {doc.metadata['page']})"
        
        formatted_docs.append(f"[SOURCE: {source}{section_info}]\n{doc.page_content}")
    
    return "\n\n".join(formatted_docs)

# Initialize QA chain on app load
@st.cache_resource(show_spinner=False)
def initialize_qa_chain():
    # Load vector store
    vector_store = load_vector_store()
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}  # Retrieve more documents initially
    )
    
    # Create a custom retriever that prioritizes community rules and deduplicates results
    def get_prioritized_documents(query):
        # Get all relevant documents
        docs = retriever.invoke(query)
        
        # Deduplicate documents
        unique_docs = []
        seen_content = set()
        
        for doc in docs:
            # Create a content fingerprint (first 100 chars should be enough to identify duplicates)
            content_fingerprint = doc.page_content[:100]
            
            # Skip if we've already seen this content
            if content_fingerprint in seen_content:
                continue
                
            # Add to seen content
            seen_content.add(content_fingerprint)
            unique_docs.append(doc)
        
        # Split into community and USSSA docs
        community_docs = [doc for doc in unique_docs if doc.metadata.get("source") == "Panama City Community"]
        usssa_docs = [doc for doc in unique_docs if doc.metadata.get("source") != "Panama City Community"]
        
        # Ensure we have at least some community docs if they exist in the database
        if not community_docs:
            try:
                # Try a more focused search on just community docs with relevance scores
                community_docs_with_scores = vector_store.similarity_search_with_relevance_scores(
                    query, 
                    k=2,
                    filter={"source": "Panama City Community"}
                )
                # Extract just the documents from the results and deduplicate
                for doc, score in community_docs_with_scores:
                    content_fingerprint = doc.page_content[:100]
                    if content_fingerprint not in seen_content:
                        seen_content.add(content_fingerprint)
                        community_docs.append(doc)
            except Exception as e:
                print(f"Error searching for community docs: {str(e)}")
        
        # Combine with community docs first
        return community_docs + usssa_docs
    
    # Create LLM
    llm = ChatOpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name=st.secrets.get("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        temperature=0,
    )
    
    # Create prompt template
    template = """Answer the question based on the following context:

{context}

IMPORTANT INSTRUCTIONS:
These questions are regarding a slow pitch softball league in Panama City. You are a specialized assistant that ONLY answers questions about this league.
1. When Panama City Community rules conflict with USSSA rules, the Panama City Community rules take precedence.
2. Clearly indicate in your answer when you're referring to Panama City Community rules vs. USSSA rules.
3. If the question is directly addressed in the rulebook, begin your answer by stating which ruleset (Panama City Community or USSSA) primarily addresses this question. Always start with Panama City Community rules over USSSA.
4. If the question is not at all related to softball, politely decline to answer and explain that you can only provide information about the softball league. Do not decline unless it is clearly off-topic.

Question: {question}
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create chain using the new LCEL (LangChain Expression Language) approach with prioritized retriever
    qa_chain = (
        {"context": lambda x: format_docs(get_prioritized_documents(x)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain, get_prioritized_documents, vector_store

# Initialize QA chain on app load
with st.spinner("Loading rulebook database..."):
    qa_chain, get_prioritized_documents, vector_store = initialize_qa_chain()

# Get user input
query = st.chat_input("Ask a question about softball rules")

if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Track start time for cost calculation
            start_time = time.time()
            
            # Get relevant documents for source information
            source_docs = get_prioritized_documents(query)
            
            # Get response from QA chain
            response = qa_chain.invoke(query)
            
            # Calculate approximate cost (very rough estimate)
            # GPT-4o mini: $0.00015/1K input tokens, $0.0006/1K output tokens
            tokens_estimate = len(query.split()) * 2 + len(response.split()) * 2
            query_cost = (len(query.split()) / 1000) * 0.00015
            response_cost = (len(response.split()) / 1000) * 0.0006
            total_cost = query_cost + response_cost
            
            # Update session state
            st.session_state.total_tokens += tokens_estimate
            st.session_state.current_cost = total_cost
            st.session_state.last_query_cost = total_cost  # Store for next rerun
            st.session_state.total_cost += total_cost
            
            # Display response
            message_placeholder.markdown(response)
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display detailed source information in an expander
            with st.expander("View Source Details"):
                st.subheader("Sources")
                
                # Group documents by source and deduplicate
                usssa_docs = []
                community_docs = []
                
                # Track seen content to avoid duplicates
                seen_content = set()
                
                for doc in source_docs:
                    # Create a content fingerprint
                    content_fingerprint = doc.page_content[:100]
                    
                    # Skip if we've already seen this content
                    if content_fingerprint in seen_content:
                        continue
                        
                    # Add to seen content
                    seen_content.add(content_fingerprint)
                    
                    # Add to appropriate list
                    if doc.metadata.get("source") == "Panama City Community":
                        community_docs.append(doc)
                    else:
                        usssa_docs.append(doc)
                
                # Display community rules first (limit to top 2)
                if community_docs:
                    st.markdown("### Panama City Community Rules")
                    for i, doc in enumerate(community_docs[:2]):  # Only show top 2
                        with st.container():
                            # Display section path if available
                            if "section_path" in doc.metadata:
                                st.markdown(f"**Section: {doc.metadata['section_path']}**")
                            st.markdown(f"**Excerpt {i+1}:**")
                            st.code(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                
                # Display USSSA rules (limit to top 3)
                if usssa_docs:
                    st.markdown("### USSSA Rulebook")
                    
                    # PDF URL for linking to specific pages
                    pdf_url = "http://cms.usssa.net/wp-content/uploads/sites/2/2025/01/usssa-slowpitch-2025-rulebook-final.pdf"
                    
                    # Display the top 3 excerpts with links
                    for i, doc in enumerate(usssa_docs[:3]):  # Only show top 3
                        with st.container():
                            page = doc.metadata.get("page", "Unknown")
                            
                            # Create a link to the specific page in the PDF
                            if page != "Unknown" and str(page).isdigit():
                                page_link = f"{pdf_url}#page={page}"
                                st.markdown(f"**[Page {page}]({page_link}), Excerpt {i+1}:**")
                            else:
                                st.markdown(f"**Page {page}, Excerpt {i+1}:**")
                                
                            st.code(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    
                    # Collect all unique page numbers from all USSSA docs (not just the displayed ones)
                    all_pages = set()
                    for doc in usssa_docs:
                        page = doc.metadata.get("page", "Unknown")
                        if page != "Unknown":
                            all_pages.add(str(page))
                    
                    # Remove pages that were already shown in the excerpts
                    shown_pages = set(str(doc.metadata.get("page", "Unknown")) for doc in usssa_docs[:3])
                    additional_pages = all_pages - shown_pages
                    
                    # Display additional relevant pages if any exist
                    if additional_pages:
                        st.markdown("**Other relevant pages:**")
                        
                        # Sort pages numerically
                        sorted_pages = sorted(additional_pages, key=lambda x: int(x) if x.isdigit() else float('inf'))
                        
                        # Create links for each page
                        page_links = []
                        for page in sorted_pages:
                            if page.isdigit():
                                page_link = f"{pdf_url}#page={page}"
                                page_links.append(f"[Page {page}]({page_link})")
                            else:
                                page_links.append(f"Page {page}")
                        
                        # Display as a comma-separated list
                        st.markdown(", ".join(page_links))
            
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
