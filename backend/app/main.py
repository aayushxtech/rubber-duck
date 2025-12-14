from fastapi import FastAPI
from app.routes.duck import router as duck_router

app = FastAPI()

app.include_router(duck_router, prefix="/api", tags=["Duck Core Service[v1]"])


@app.get("/health", tags=["Health"])
async def read_root():
    return {"status": "ok"}
