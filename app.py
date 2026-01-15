import gradio as gr
import os
import json

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === OPBOUW VECTOR STORE ===
print("Erfocentrum Wegwijzer wordt geladen...")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def beantwoord_vraag(message, history):
    """Beantwoord een vraag met de RAG chatbot."""
    if not message.strip():
        return "Stel gerust een vraag over erfelijkheid!"
    
    docs = retriever.invoke(message)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    bronnen = set()
    for doc in docs:
        bronnen.add(doc.metadata.get('source', ''))
    
    prompt = f"""Je bent de Erfocentrum Wegwijzer. Beantwoord de vraag op basis van de context.
Als het antwoord niet in de context staat, zeg dat je het niet weet en verwijs naar de Erfolijn (https://www.erfelijkheid.nl/contact).
Geef geen persoonlijk medisch advies.

Context:
{context}

Vraag: {message}

Antwoord in het Nederlands:"""
    
    response = llm.invoke(prompt)
    antwoord = response.content
    
    if bronnen:
        antwoord += "\n\n📚 **Bronnen:**\n"
        for bron in list(bronnen)[:3]:
            if bron:
                antwoord += f"- {bron}\n"
    
    return antwoord


# === INTERFACE MET PRIVACY FLOW ===

def naar_privacy():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def akkoord_gegeven():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def niet_akkoord_gegeven():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value="""
## Je hebt niet akkoord gegeven

Je kunt de chatbot alleen gebruiken als je akkoord gaat met de privacyverklaring.

Bezoek [erfelijkheid.nl](https://www.erfelijkheid.nl) voor meer informatie over erfelijkheid.
""")


with gr.Blocks(title="Erfocentrum Wegwijzer") as demo:
    
    # === WELKOM SECTIE ===
    with gr.Column(visible=True) as welkom_sectie:
        gr.Markdown("""
# 🧬 Erfocentrum Wegwijzer

**Stel je vraag aan de Wegwijzer van het Erfocentrum.**

Ik help je graag met het zoeken naar algemene informatie over erfelijke ziektes of aandoeningen.

⚠️ **Let op:** voor een persoonlijk medisch advies kan je het beste contact opnemen met je (huis)arts.
""")
        start_btn = gr.Button("▶️ Start chat", variant="primary", size="lg")
    
    # === PRIVACY SECTIE ===
    with gr.Column(visible=False) as privacy_sectie:
        gr.Markdown("""
## Privacy

Wij vinden jouw privacy heel belangrijk. Bekijk daarom onze [privacyverklaring](https://www.erfelijkheid.nl/privacy).

Je gesprek wordt **niet opgeslagen** na het sluiten van deze chat.

**Ga je hiermee akkoord?**
""")
        with gr.Row():
            akkoord_btn = gr.Button("✅ Akkoord", variant="primary")
            niet_akkoord_btn = gr.Button("❌ Niet akkoord", variant="secondary")
    
    # === CHAT SECTIE ===
    with gr.Column(visible=False) as chat_sectie:
        gr.Markdown("## 🧬 Erfocentrum Wegwijzer\n\nDank voor je akkoord! Waar ben je naar op zoek?")
        
        chatbot = gr.ChatInterface(
            fn=beantwoord_vraag,
            examples=[
                "Wat is erfelijkheid?",
                "Hoe werkt DNA-onderzoek?",
                "Is kanker erfelijk?",
                "Is dementie erfelijk?"
            ]
        )
        
        gr.Markdown("""
---
<small>De antwoorden zijn gebaseerd op informatie van [erfelijkheid.nl](https://www.erfelijkheid.nl), 
zorgvuldig samengesteld en gecontroleerd door medici die aangesloten zijn bij het Erfocentrum.</small>
""")
    
    # === NIET AKKOORD SECTIE ===
    with gr.Column(visible=False) as niet_akkoord_sectie:
        niet_akkoord_tekst = gr.Markdown("")
    
    # Event handlers
    start_btn.click(
        fn=naar_privacy,
        outputs=[welkom_sectie, privacy_sectie, chat_sectie]
    )
    
    akkoord_btn.click(
        fn=akkoord_gegeven,
        outputs=[welkom_sectie, privacy_sectie, chat_sectie]
    )
    
    niet_akkoord_btn.click(
        fn=niet_akkoord_gegeven,
        outputs=[welkom_sectie, privacy_sectie, niet_akkoord_sectie]
    )


if __name__ == "__main__":
    demo.launch()
