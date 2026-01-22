import gradio as gr
import os
import json
import time
import openai
import chromadb
from chromadb.utils import embedding_functions

# === CONFIGURATIE ===
print("Erfocentrum Wegwijzer wordt geladen...")

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = chroma_client.get_collection(
        name="erfocentrum",
        embedding_function=openai_ef
    )
    print(f"Bestaande collectie geladen met {collection.count()} documenten")
except:
    print("Nieuwe collectie aanmaken...")
    collection = chroma_client.create_collection(
        name="erfocentrum",
        embedding_function=openai_ef
    )
    
    with open("erfocentrum_data.json", "r", encoding="utf-8") as f:
        scraped_pages = json.load(f)
    
    filtered_pages = [p for p in scraped_pages if p.get('content', '').strip() and len(p['content']) > 200][:300]
    print(f"Gebruik {len(filtered_pages)} pagina's")
    
    documents, metadatas, ids = [], [], []
    
    for i, page in enumerate(filtered_pages):
        content = page['content']
        for j in range(0, len(content), 1300):
            chunk = content[j:j + 1500]
            if len(chunk) > 100:
                documents.append(chunk)
                metadatas.append({'source': page['url'], 'title': page['title']})
                ids.append(f"doc_{i}_{j}")
    
    print(f"Totaal {len(documents)} chunks")
    
    for i in range(0, len(documents), 50):
        end_idx = min(i + 50, len(documents))
        try:
            collection.add(documents=documents[i:end_idx], metadatas=metadatas[i:end_idx], ids=ids[i:end_idx])
            print(f"Toegevoegd: {end_idx}/{len(documents)}")
            time.sleep(1)
        except:
            time.sleep(5)
    
    print(f"Collectie aangemaakt met {collection.count()} chunks")

print("Vector store klaar!")


# === CHATBOT FUNCTIE - alleen message, geen history ===
def beantwoord_vraag(message, history):
    if not message.strip():
        return "Stel gerust een vraag over erfelijkheid."
    
    results = collection.query(query_texts=[message], n_results=5)
    
    context_parts, bronnen = [], []
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            context_parts.append(doc)
            if results['metadatas'] and results['metadatas'][0]:
                bron = results['metadatas'][0][i].get('source', '')
                if bron and bron not in bronnen:
                    bronnen.append(bron)
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Je bent de Erfocentrum Wegwijzer, een professionele assistent over erfelijkheid en genetica.

REGELS:
1. Beantwoord alleen op basis van de context hieronder
2. Als het antwoord niet in de context staat, verwijs naar de Erfolijn
3. Geen persoonlijk medisch advies - verwijs naar een arts
4. Antwoord in het Nederlands, duidelijk en bondig
5. Professionele toon zonder emoji's

CONTEXT:
{context}

VRAAG: {message}

Antwoord:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    antwoord = response.choices[0].message.content
    
    if any(kw in antwoord.lower() for kw in ['niet gevonden', 'geen informatie', 'niet in de context', 'weet ik niet']):
        if 'erfolijn' not in antwoord.lower():
            antwoord += "\n\nNeem contact op met de Erfolijn: https://www.erfelijkheid.nl/contact"
    elif bronnen:
        antwoord += "\n\nMeer informatie:\n" + "\n".join([f"• {b}" for b in bronnen[:2]])
    
    return antwoord


# === CSS ===
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', -apple-system, sans-serif !important;
    max-width: 600px !important;
    margin: 0 auto !important;
    background: #F5FAFA !important;
}
footer { display: none !important; }

.chat-header {
    background: linear-gradient(135deg, #00B4C5 0%, #008B9A 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 0 0 24px 24px;
    text-align: center;
    margin-bottom: 1rem;
}
.chat-header h1 { margin: 0; font-size: 1.4rem; }
.chat-header p { margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9; }

.avatar-container {
    width: 60px; height: 60px;
    background: white;
    border-radius: 50%;
    margin: 0 auto 0.75rem auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
}

.welcome-box {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem;
    box-shadow: 0 2px 12px rgba(0, 165, 181, 0.12);
}
.welcome-box p { color: #444; line-height: 1.6; margin: 0; }

.info-box {
    background: #E8F6F7;
    border-radius: 16px;
    padding: 1rem;
    margin: 1rem;
    border-left: 4px solid #00A5B5;
}
.info-box p { color: #333; margin: 0; font-size: 0.9rem; }
.info-box a { color: #008B9A; }

.footer-text {
    text-align: center;
    color: #888;
    font-size: 0.8rem;
    padding: 1rem;
}
.footer-text a { color: #008B9A; text-decoration: none; }
"""


# === INTERFACE FUNCTIES ===
def naar_privacy():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def akkoord_gegeven():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def niet_akkoord_gegeven():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


# === APP ===
with gr.Blocks(css=custom_css, title="Erfocentrum Wegwijzer") as demo:
    
    # WELKOM
    with gr.Column(visible=True) as welkom_sectie:
        gr.HTML("""
        <div class="chat-header">
            <div class="avatar-container">🧬</div>
            <h1>Erfocentrum Wegwijzer</h1>
            <p>Chat met onze digitale assistent</p>
        </div>
        <div class="welcome-box">
            <p>Stel je vraag aan de Wegwijzer van het Erfocentrum. Ik help je graag met het zoeken naar algemene informatie over erfelijke ziektes of aandoeningen.</p>
        </div>
        <div class="info-box">
            <p><strong>Let op:</strong> voor persoonlijk medisch advies kun je het beste contact opnemen met je (huis)arts.</p>
        </div>
        """)
        start_btn = gr.Button("Start chat", variant="primary")
    
    # PRIVACY
    with gr.Column(visible=False) as privacy_sectie:
        gr.HTML("""
        <div class="chat-header">
            <div class="avatar-container">🔒</div>
            <h1>Privacy</h1>
            <p>Jouw gegevens zijn veilig</p>
        </div>
        <div class="info-box">
            <p>Wij vinden jouw privacy heel belangrijk. Bekijk onze <a href="https://www.erfelijkheid.nl/privacy" target="_blank">privacyverklaring</a>.</p>
        </div>
        <div class="welcome-box">
            <p>Je gesprek wordt <strong>niet opgeslagen</strong> na het sluiten van deze chat.</p>
        </div>
        """)
        gr.Markdown("**Ga je akkoord met de privacyverklaring?**")
        with gr.Row():
            akkoord_btn = gr.Button("Akkoord", variant="primary")
            niet_akkoord_btn = gr.Button("Niet akkoord", variant="secondary")
    
    # CHAT - gebruik gr.ChatInterface
    with gr.Column(visible=False) as chat_sectie:
        gr.HTML("""
        <div class="chat-header">
            <div class="avatar-container">🧬</div>
            <h1>Erfocentrum Wegwijzer</h1>
            <p>Stel je vraag over erfelijkheid</p>
        </div>
        """)
        
        gr.ChatInterface(
            fn=beantwoord_vraag,
            examples=["Wat is erfelijkheid?", "Is kanker erfelijk?", "Hoe werkt DNA-onderzoek?"],
        )
        
        gr.HTML("""
        <div class="footer-text">
            Gebaseerd op <a href="https://www.erfelijkheid.nl" target="_blank">erfelijkheid.nl</a> · 
            <a href="https://www.erfelijkheid.nl/contact" target="_blank">Contact Erfolijn</a>
        </div>
        """)
    
    # NIET AKKOORD
    with gr.Column(visible=False) as niet_akkoord_sectie:
        gr.HTML("""
        <div class="chat-header">
            <div class="avatar-container">🧬</div>
            <h1>Erfocentrum Wegwijzer</h1>
        </div>
        <div class="welcome-box">
            <p><strong>Je hebt niet akkoord gegeven</strong></p>
            <p>Je kunt de chatbot alleen gebruiken met akkoord op de privacyverklaring.</p>
            <p>Bezoek <a href="https://www.erfelijkheid.nl">erfelijkheid.nl</a> of de <a href="https://www.erfelijkheid.nl/contact">Erfolijn</a>.</p>
        </div>
        """)
    
    # Events
    start_btn.click(fn=naar_privacy, outputs=[welkom_sectie, privacy_sectie, chat_sectie, niet_akkoord_sectie])
    akkoord_btn.click(fn=akkoord_gegeven, outputs=[welkom_sectie, privacy_sectie, chat_sectie, niet_akkoord_sectie])
    niet_akkoord_btn.click(fn=niet_akkoord_gegeven, outputs=[welkom_sectie, privacy_sectie, chat_sectie, niet_akkoord_sectie])

if __name__ == "__main__":
    demo.launch()
