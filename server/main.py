import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# Definindo o modelo de input do usuário
class UserInput(BaseModel):
    message: str

# Inicializando o cliente Groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Inicializando o histórico do chat
system_prompt = {
    "role": "system",
    "content": "you are an AI Assistant and your name is AIrton. Your main job is to show the entry and exit, like you're a tourist guide. Remember to reply in the same language that the customer is using"
}

# Criar o FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Defina a origem do seu frontend Angular
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os cabeçalhos
)


# Histórico de chat pode ser armazenado em memória ou banco de dados se necessário
chat_history = [system_prompt]

@app.post("/chat/")
async def chat_with_ai(user_input: UserInput):
    try:
        # Adicionar a entrada do usuário ao histórico
        chat_history.append({"role": "user", "content": user_input.message})

        # Gerar resposta do chatbot
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=chat_history,
            max_tokens=1000,
            temperature=1.2
        )

        # Adicionar a resposta ao histórico
        assistant_response = response.choices[0].message.content
        chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })

        # Retornar a resposta do chatbot
        return {"assistant_message": assistant_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
