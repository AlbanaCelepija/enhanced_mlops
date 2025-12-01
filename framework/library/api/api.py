import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from library.src.artifact_types import Report


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    message: str
    usecase: str


@app.post("/run_operation")
def run_operation(operation_id: str = "is_prod_reusable", params: Message = "nlp"):
    return "OK!"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    #report = Report("library/use_cases/nlp/src/local_platform/data/people_data.json")
    #frame = report.load_report()
    #print(frame)
