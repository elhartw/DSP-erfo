# ================================
# Erfocentrum Website Chatbot (Verbeterd)
# ================================
# Run met:
#   python erfocentrum-chatbot.py
# ================================

import os
import json
import panel as pn
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

# ================================================
# CONFIGURATIE - PAS HIER HET PAD AAN
# ================================================
JSON_FILE_PATH = "erfocentrum_data.json"
# ================================================

# ------------------------------------------------
# Laad environment variables
# ------------------------------------------------
load_dotenv()

if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = "ErfocentrumChatbot/1.0"

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "⚠️ OPENAI_API_KEY niet gevonden!\n"
        "Maak een .env bestand in dezelfde folder met:\n"
        "OPENAI_API_KEY=sk-jouw-api-key-hier"
    )

print(f"✅ API key geladen: {os.getenv('OPENAI_API_KEY')[:20]}...")

# ------------------------------------------------
# Panel setup
# ------------------------------------------------
pn.extension()

# ------------------------------------------------
# Globale variabelen
# ------------------------------------------------
website_chunks = []
llm = None
page_sources = {}
chat_history = []  # 🆕 Conversatie geschiedenis

# ------------------------------------------------
# Widgets
# ------------------------------------------------
k_slider = pn.widgets.IntSlider(
    name="Aantal tekstchunks",
    start=1,
    end=8,
    step=1,
    value=4
)

temperature_slider = pn.widgets.FloatSlider(
    name="Creativiteit (temperature)",
    start=0.0,
    end=1.0,
    step=0.1,
    value=0.0
)

clear_history_btn = pn.widgets.Button(
    name="🗑️ Wis Gesprek",
    button_type="warning"
)

export_btn = pn.widgets.Button(
    name="📥 Exporteer Chat",
    button_type="primary"
)

status_text = pn.pane.Markdown("⏳ Website data wordt geladen...")

chat_input = pn.widgets.TextInput(
    placeholder="Even geduld, data wordt geladen..."
)

# 🆕 Voorgestelde vragen
suggested_questions = [
    "Wat is erfelijkheid?",
    "Is kanker erfelijk?",
    "Wat doet het Erfocentrum?",
    "Hoe werkt DNA-onderzoek?",
]

suggestion_buttons = pn.Column(
    pn.pane.Markdown("**💡 Voorgestelde vragen:**"),
    pn.Row(
        *[pn.widgets.Button(name=q, button_type="light", width=200) for q in suggested_questions[:2]]
    ),
    pn.Row(
        *[pn.widgets.Button(name=q, button_type="light", width=200) for q in suggested_questions[2:]]
    ),
    sizing_mode="stretch_width"
)

# ------------------------------------------------
# JSON data laden
# ------------------------------------------------
def load_json_data():
    global website_chunks, llm, page_sources
    
    try:
        if not os.path.exists(JSON_FILE_PATH):
            raise FileNotFoundError(
                f"❌ JSON bestand niet gevonden: {JSON_FILE_PATH}\n"
                f"Zorg dat het bestand in dezelfde folder staat als dit script."
            )
        
        print(f"📁 JSON bestand laden: {JSON_FILE_PATH}")
        
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("❌ JSON moet een lijst zijn van pagina objecten!")
        
        print(f"✅ {len(data)} pagina's gevonden in JSON")
        
        all_docs = []
        total_chars = 0
        
        for page in data:
            if not page.get('success', False):
                continue
                
            content = page.get('content', '')
            url = page.get('url', 'Onbekend')
            title = page.get('title', 'Geen titel')
            
            if content and len(content) > 50:
                total_chars += len(content)
                
                class SimpleDoc:
                    def __init__(self, content, url, title):
                        self.page_content = content
                        self.metadata = {"source": url, "title": title}
                
                all_docs.append(SimpleDoc(content, url, title))
        
        if not all_docs:
            raise ValueError("❌ Geen bruikbare content gevonden in JSON!")
        
        print(f"✅ {len(all_docs)} pagina's met {total_chars:,} karakters verwerkt")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        website_chunks = splitter.split_documents(all_docs)
        
        page_sources = {}
        for chunk in website_chunks:
            source = chunk.metadata.get('source', 'Onbekend')
            title = chunk.metadata.get('title', 'Geen titel')
            page_sources[id(chunk)] = {'source': source, 'title': title}
        
        print(f"✅ {len(website_chunks)} tekstchunks gemaakt")
        
        # LLM met dynamische temperature
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature_slider.value,
            streaming=True  # 🆕 Streaming enabled
        )
        
        print("✅ LLM geïnitialiseerd")
        
        status_text.object = f"✅ Data geladen! {len(website_chunks)} chunks van {len(all_docs)} pagina's beschikbaar."
        chat_input.placeholder = "Stel je vraag over het Erfocentrum..."
        
        return True, len(all_docs), len(website_chunks), total_chars
        
    except Exception as e:
        error_msg = f"❌ Error bij laden: {str(e)}"
        status_text.object = error_msg
        print(error_msg)
        return False, 0, 0, 0

# ------------------------------------------------
# 🆕 Verbeterde zoekfunctie met TF-IDF-achtige scoring
# ------------------------------------------------
def search_relevant_chunks(query, chunks, k=4):
    """Zoek relevante chunks met betere scoring"""
    query_words = set(query.lower().split())
    query_lower = query.lower()
    
    scored_chunks = []
    for chunk in chunks:
        content_lower = chunk.page_content.lower()
        chunk_words = set(content_lower.split())
        
        score = 0
        
        # 1. Exacte match (hoogste prioriteit)
        if query_lower in content_lower:
            score += 50
        
        # 2. Overlappende woorden
        overlap = query_words.intersection(chunk_words)
        score += len(overlap) * 2
        
        # 3. Bonus voor langere woorden (belangrijker)
        for word in overlap:
            if len(word) > 5:
                score += 3
        
        # 4. Bonus voor woorden aan begin van chunk (vaak belangrijker)
        first_100 = content_lower[:100]
        for word in query_words:
            if word in first_100:
                score += 2
        
        # 5. Percentage overlap
        if len(chunk_words) > 0:
            overlap_percentage = len(overlap) / len(query_words)
            score += overlap_percentage * 10
        
        scored_chunks.append((score, chunk))
    
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:k] if score > 0]

# ------------------------------------------------
# 🆕 Chat callback met conversatie geschiedenis
# ------------------------------------------------
async def respond(message, user, chat):
    global website_chunks, llm, chat_history
    
    if not website_chunks or llm is None:
        chat.send(
            {"user": "System", "value": "⚠️ De website data kon niet worden geladen. Check de console voor details."},
            respond=False
        )
        return
    
    try:
        # Zoek relevante chunks
        relevant_chunks = search_relevant_chunks(message, website_chunks, k=k_slider.value)
        
        if not relevant_chunks:
            yield {"user": "Erfocentrum Bot", "value": "❌ Geen relevante informatie gevonden. Probeer een andere vraag."}
            return
        
        # Maak context
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
        # 🆕 Voeg chat history toe aan prompt
        history_text = ""
        if chat_history:
            history_text = "\n\nEerdere conversatie:\n"
            for msg in chat_history[-4:]:  # Laatste 4 berichten
                role = "Gebruiker" if isinstance(msg, HumanMessage) else "Assistent"
                history_text += f"{role}: {msg.content}\n"
        
        system_prompt = (
            "Je bent een behulpzame assistent van het Erfocentrum. "
            "Beantwoord vragen op basis van de informatie van de Erfocentrum website. "
            "Gebruik ALLEEN de volgende context om de vraag te beantwoorden. "
            "Als je het antwoord niet weet op basis van de gegeven context, zeg dan eerlijk dat je het niet weet. "
            "Wees vriendelijk en professioneel in je antwoorden."
            f"{history_text}\n\n"
            f"Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Update LLM temperature
        llm.temperature = temperature_slider.value
        
        chain = prompt | llm | StrOutputParser()
        
        # 🆕 Streaming response
        full_response = ""
        output = pn.Column(sizing_mode="stretch_width")
        response_text = pn.pane.Markdown("", sizing_mode="stretch_width")
        output.append(response_text)
        
        # Toon direct de output container
        yield {"user": "Erfocentrum Bot", "value": output}
        
        # Stream het antwoord
        for chunk in chain.stream({"input": message}):
            full_response += chunk
            response_text.object = full_response
        
        # Voeg bronnen toe
        output.append(pn.layout.Divider())
        output.append(f"**📚 Bronnen ({len(relevant_chunks)} chunks):**")
        
        shown_sources = set()
        for chunk in relevant_chunks:
            title = chunk.metadata.get('title', 'Geen titel')
            source = chunk.metadata.get('source', 'Onbekend')
            if source not in shown_sources:
                output.append(f"- [{title}]({source})")
                shown_sources.add(source)
        
        # 🆕 Voeg feedback knoppen toe
        feedback_row = pn.Row(
            pn.widgets.Button(name="👍 Nuttig", button_type="success", width=100),
            pn.widgets.Button(name="👎 Niet nuttig", button_type="danger", width=100),
        )
        output.append(pn.layout.Divider())
        output.append(feedback_row)
        
        # 🆕 Update chat history
        chat_history.append(HumanMessage(content=message))
        chat_history.append(AIMessage(content=full_response))
        
    except Exception as e:
        yield {"user": "Error", "value": f"❌ Query error: {str(e)}"}

# ------------------------------------------------
# Chat UI
# ------------------------------------------------
chat_interface = pn.chat.ChatInterface(
    callback=respond,
    widgets=[chat_input],
    sizing_mode="stretch_width",
    help_text="Stel vragen over het Erfocentrum!"
)

# ------------------------------------------------
# 🆕 Button callbacks
# ------------------------------------------------
def clear_history(event):
    global chat_history
    chat_history = []
    chat_interface.clear()
    chat_interface.send(
        {"user": "System", "value": "✅ Gesprek gewist! Je kunt een nieuw gesprek beginnen."},
        respond=False
    )

def export_chat(event):
    """Exporteer chat geschiedenis naar TXT bestand"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_export_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("ERFOCENTRUM CHATBOT - GESPREK EXPORT\n")
        f.write(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, msg in enumerate(chat_history):
            role = "👤 Gebruiker" if isinstance(msg, HumanMessage) else "🤖 Erfocentrum Bot"
            f.write(f"{role}:\n{msg.content}\n\n")
            f.write("-" * 50 + "\n\n")
    
    chat_interface.send(
        {"user": "System", "value": f"✅ Chat geëxporteerd naar: `{filename}`"},
        respond=False
    )

def ask_suggestion(event):
    """Verstuur voorgestelde vraag"""
    question = event.obj.name
    chat_interface.send(question, user="Gebruiker", respond=True)

clear_history_btn.on_click(clear_history)
export_btn.on_click(export_chat)

for btn in suggestion_buttons[1].objects + suggestion_buttons[2].objects:
    btn.on_click(ask_suggestion)

# ------------------------------------------------
# Layout
# ------------------------------------------------
template = pn.template.BootstrapTemplate(
    title="🏢 Erfocentrum Chatbot",
    sidebar=[
        pn.pane.Markdown("## ⚙️ Instellingen"),
        k_slider,
        temperature_slider,
        pn.layout.Divider(),
        pn.pane.Markdown("## 🔧 Acties"),
        clear_history_btn,
        export_btn,
        pn.layout.Divider(),
        pn.pane.Markdown("## 📊 Status"),
        status_text,
    ],
    main=[
        pn.pane.Markdown("# Welkom bij de Erfocentrum Chatbot! 👋"),
        pn.pane.Markdown("Stel vragen over erfelijkheid en gezondheid."),
        suggestion_buttons,
        pn.layout.Divider(),
        chat_interface
    ],
)

# ------------------------------------------------
# Startup
# ------------------------------------------------
def startup():
    success, num_pages, num_chunks, total_chars = load_json_data()
    
    if success:
        chat_interface.send(
            {"user": "System", "value": f"✅ Website data geladen!\n\n📊 **Statistieken:**\n- {num_pages} pagina's\n- {num_chunks} tekstchunks\n- {total_chars:,} karakters\n\n💡 Klik op een voorgestelde vraag of stel je eigen vraag!"},
            respond=False
        )
    else:
        chat_interface.send(
            {"user": "System", "value": f"❌ Kon website data niet laden. Check of het bestand '{JSON_FILE_PATH}' bestaat."},
            respond=False
        )

pn.state.onload(startup)

# ------------------------------------------------
# Serve
# ------------------------------------------------
if __name__ == "__main__":
    pn.serve(
        template,
        title="Erfocentrum Chatbot",
        show=True,
        port=5006
    )