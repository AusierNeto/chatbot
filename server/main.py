import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

class UserInput(BaseModel):
    message: str

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

system_prompt = {
    "role": "system",
    "content": "you are an AI Assistant and your name is AIrton. Your main job is to show the entry and exit, like you're a tourist guide. Remember to reply in the same language that the customer is using"
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

chat_history = [system_prompt]

@app.post("/chat/")
async def chat_with_ai(user_input: UserInput):
    try:    
        chat_history.append({"role": "user", "content": user_input.message})

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=chat_history,
            max_tokens=1000,
            temperature=1.2
        )
    
        assistant_response = response.choices[0].message.content
        chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })
    
        return {"assistant_message": assistant_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
