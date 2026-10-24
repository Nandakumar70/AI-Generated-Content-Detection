from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Import your routers
from Backend.image_detector import router as image_router
from Backend.text_detector import router as text_router

app = FastAPI(title="VeritasAI Forensic API")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Connect the endpoints
app.include_router(image_router)
app.include_router(text_router)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()