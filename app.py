import gradio as gr
import os
import json

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Laad data en bouw vector store
print("App wordt geladen...")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Check of vector store al bestaat
if os.path.exists("./chroma_db") and len(os.listdir("./chroma_db")) > 0:
    print("Vector store laden...")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
else:
    print("Vector store bouwen vanuit data...")
    with open("erfocentrum_data.json", "r", encoding="utf-8") as f:
        scraped_pages = json.load(f)
    
    documents = []
    for page in scraped_pages:
        if page.get('content', '').strip():
            doc = Document(
                page_content=page['content'],
                metadata={'source': page['url'], 'title': page['title']}
            )
            documents.append(doc)
    
    print(f"Aantal documenten: {len(documents)}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Aantal chunks: {len(chunks)}")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

print("Vector store klaar!")

# Maak retriever en LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def chat(message, history):
    """Beantwoord een vraag met de RAG chatbot."""
    if not message.strip():
        return "Stel gerust een vraag over erfelijkheid!"
    
    # Haal relevante documenten op
    docs = retriever.invoke(message)
    
    # Bouw context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Maak prompt
    prompt = f"""Je bent een behulpzame assistent die vragen beantwoordt over erfelijkheid en genetica.
Gebruik ALLEEN de volgende context om de vraag te beantwoorden. Als je het antwoord niet weet op basis van de context, zeg dat dan eerlijk.

Context:
{context}

Vraag: {message}

Antwoord in het Nederlands:"""
    
    # Genereer antwoord
    response = llm.invoke(prompt)
    answer = response.content
    
    # Voeg bronnen toe
    sources = set()
    for doc in docs:
        sources.add(doc.metadata.get('source', ''))
    
    if sources:
        answer += "\n\n📚 **Bronnen:**\n"
        for source in list(sources)[:3]:
            if source:
                answer += f"- {source}\n"
    
    return answer

# Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    title="🧬 Erfocentrum Chatbot",
    description="Stel vragen over erfelijkheid, DNA en genetische aandoeningen.",
    examples=[
        "Wat is erfelijkheid?",
        "Hoe werkt DNA-onderzoek?",
        "Is kanker erfelijk?"
    ]
)

if __name__ == "__main__":
    demo.launch()
