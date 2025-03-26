from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8000",  # For local development (React/Vue/Next/etc.)
        ],
        allow_credentials=True,  # Allow cookies, Authorization headers
        allow_methods=["GET", "POST"],  # Limit methods to what you use
        allow_headers=["Authorization", "Content-Type"],  # Limit to necessary headers
    )
