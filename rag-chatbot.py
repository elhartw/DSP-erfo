# ================================
# RAG Chatbot – LangChain NIEUWE API
# ================================
# Run met:
#   python rag-chatbot.py
# ================================

import os
import tempfile
import panel as pn

# Nieuwe LangChain importsp
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------
# Panel setup
# ------------------------------------------------
pn.extension()

# ------------------------------------------------
# Widgets
# ------------------------------------------------
pdf_input = pn.widgets.FileInput(accept=".pdf", height=50)

key_input = pn.widgets.PasswordInput(
    name="OpenAI API Key",
    placeholder="sk-..."
)

k_slider = pn.widgets.IntSlider(
    name="Aantal relevante tekstchunks (k)",
    start=1,
    end=5,
    step=1,
    value=2
)

chat_input = pn.widgets.TextInput(
    placeholder="Upload eerst een PDF..."
)

# ------------------------------------------------
# RAG chain initialisatie (NIEUWE API)
# ------------------------------------------------
def initialize_chain():
    if not key_input.value:
        raise ValueError("Geen OpenAI API key ingevoerd")

    os.environ["OPENAI_API_KEY"] = key_input.value

    if not pdf_input.value:
        return None

    # PDF tijdelijk opslaan
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_input.value)
        pdf_path = f.name

    try:
        # PDF laden
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Tekst splitsen
        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )
        texts = splitter.split_documents(documents)

        # Embeddings + vector store
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(texts, embeddings)

        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_slider.value}
        )

        # LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        # NIEUWE MANIER: Prompt template
        system_prompt = (
            "Je bent een behulpzame assistent die vragen beantwoordt op basis van de gegeven context. "
            "Gebruik de volgende stukken context om de vraag te beantwoorden. "
            "Als je het antwoord niet weet, zeg dan gewoon dat je het niet weet.\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # NIEUWE MANIER: Chain opbouwen
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        return rag_chain
    
    finally:
        # Cleanup tijdelijk bestand
        try:
            os.unlink(pdf_path)
        except:
            pass

# ------------------------------------------------
# Chat callback
# ------------------------------------------------
async def respond(message, user, chat):
    if not pdf_input.value:
        chat.send(
            {"user": "System", "value": "⚠️ Upload eerst een PDF."},
            respond=False
        )
        return

    try:
        rag_chain = initialize_chain()
        if rag_chain is None:
            chat.send(
                {"user": "System", "value": "⚠️ Kon geen chain initialiseren."},
                respond=False
            )
            return
    except Exception as e:
        chat.send(
            {"user": "Error", "value": f"Error: {str(e)}"},
            respond=False
        )
        return

    try:
        # RAG query uitvoeren (LET OP: 'input' in plaats van 'query')
        response = rag_chain.invoke({"input": message})

        # LET OP: 'answer' in plaats van 'result'
        output = pn.Column(response["answer"], sizing_mode="stretch_width")
        output.append(pn.layout.Divider())

        # Bronnen tonen (nu in 'context' in plaats van 'source_documents')
        if "context" in response:
            for doc in response["context"][::-1]:
                page = doc.metadata.get("page", "?")
                output.append(f"**Pagina {page}**")
                output.append(f"```\n{doc.page_content}\n```")

        yield {"user": "OpenAI", "value": output}
        
    except Exception as e:
        yield {"user": "Error", "value": f"Query error: {str(e)}"}

# ------------------------------------------------
# Chat UI
# ------------------------------------------------
chat_interface = pn.chat.ChatInterface(
    callback=respond,
    widgets=[pdf_input, chat_input],
    sizing_mode="stretch_width",
    help_text="Upload een PDF en stel vragen!"
)

chat_interface.send(
    {"user": "System", "value": "Upload een PDF om te beginnen."},
    respond=False,
)

# ------------------------------------------------
# Layout
# ------------------------------------------------
template = pn.template.BootstrapTemplate(
    title="📄 RAG Chatbot (LangChain + Panel)",
    sidebar=[
        pn.pane.Markdown("## Instellingen"),
        key_input,
        k_slider,
    ],
    main=[chat_interface],
)

# ------------------------------------------------
# Serve app
# ------------------------------------------------
pn.serve(
    template,
    title="RAG Chatbot",
    show=True,
    port=5006
)