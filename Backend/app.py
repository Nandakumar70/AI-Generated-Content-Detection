from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Backend.image_detector import router as image_detector_router

# Initialize the main FastAPI application
app = FastAPI(title="AI Image Detection API")

# Setup CORS so your frontend can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Connect the image detection endpoints
app.include_router(image_detector_router)

@app.get("/")
def read_root():
    return {"message": "AI Image Detection API is running. Go to /docs to test it!"}