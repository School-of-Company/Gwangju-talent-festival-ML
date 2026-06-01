from fastapi import FastAPI

app = FastAPI(title="Gwangju Talent Festival ML")

@app.get("/health")
def health():
    return {"status": "ok"}
