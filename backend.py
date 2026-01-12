from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Sta je homepage (Live Server) toe om de API te callen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later beperken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    # Dummy antwoord (later vervangen door RAG)
    return {"answer": f"Dit is een test-antwoord. Jij vroeg: {req.message}"}
