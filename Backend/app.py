from fastapi import FastAPI

app = FastAPI(
    title="AI-Generated Content Detection API",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "project": "AI-Generated Content Detection",
        "status": "Backend Running Successfully 🚀"
    }