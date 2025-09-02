# Import necessary libraries and modules
from openai import OpenAI
from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
from typing import Annotated
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
import os
from dotenv import load_dotenv

load_dotenv()

# Create an instance of the FastAPI application
app = FastAPI()

# Get the directory of the current script to find templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Initialize a single OpenAI client for both chat and image generation with OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_SECRET_KEY'),
)

# NOTE: For a single-user demo, this is fine, but for multiple users, each session
# needs its own chat log.
chat_log = [{
    'role': 'system',
    'content': (
        "You are a Python tutor AI, completely dedicated to teaching users "
        "Python from scratch. Provide clear instructions on Python concepts, "
        "best practices, and syntax. Help create a path of learning so users "
        "can build real-life, production-ready Python applications."
    )
}]

# A separate list to store just the chat content for the template.
chat_responses = []

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Handles the GET request for the home page (chat page)."""
    return templates.TemplateResponse(
        "chat.html", {"request": request, "chat_responses": chat_responses}
    )

@app.get("/style.css", response_class=FileResponse)
async def serve_css():
    """Serves the static CSS file."""
    return FileResponse(os.path.join(BASE_DIR, "templates/style.css"))

@app.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    """Handles WebSocket connections for real-time chat."""
    await websocket.accept()

    # Initial greeting
    greeting = "Hello! I'm your Python tutor AI. How can I help you learn Python today?"
    await websocket.send_text(greeting)

    try:
        while True:
            # Receive user input
            user_input = await websocket.receive_text()
            chat_log.append({'role': 'user', 'content': user_input})

            # Call the OpenRouter API for chat completions (streaming)
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://127.0.0.1:8000/",
                    "X-Title": "Python Tutor AI",
                },
                model="openai/gpt-3.5-turbo",
                messages=chat_log,
                temperature=0.6,
                stream=True
            )

            ai_response = ""
            for chunk in completion:
                # ✅ Corrected: ChoiceDelta object, access `.content` directly
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    await websocket.send_text(content)
                    ai_response += content

            # Save AI response in logs
            chat_log.append({'role': 'assistant', 'content': ai_response})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/image", response_class=HTMLResponse)
async def image_page(request: Request):
    """Handles the GET request for the image generation page."""
    return templates.TemplateResponse("image.html", {"request": request})

@app.post("/image", response_class=HTMLResponse)
async def create_image(request: Request, user_input: Annotated[str, Form()]):
    """Handles the POST request for image generation using the OpenRouter API."""
    try:
        # Correct method for image generation using client.images.generate
        response = client.images.generate(
            model="openai/dall-e-3",
            prompt=user_input,
            n=1,
            size="1024x1024",
        )
        
        # Correct way to access the image URL from the response
        image_url = response.data[0].url
        
        if not image_url:
            raise ValueError("No image URL found in the API response.")

        return templates.TemplateResponse(
            "image.html", {"request": request, "image_url": image_url}
        )
    except Exception as e:
        print(f"Error generating image: {e}")
        return templates.TemplateResponse(
            "image.html",
            {
                "request": request,
                "error_message": "An error occurred. Please try a different prompt or check the OpenRouter model's response format."
            }
        )