import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,  # Now this will work!
        port=settings.PORT,  # Now this will work!
        reload=settings.DEBUG
    )