from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle5 as pickle
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from preprocess import handle_preprocess_dataframe, handle_preprocess_single_text

print("test")

with open('models/tokenizer_full_data.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model("models/RNN_LSTML_UNDERSAMPLING_LAST.h5")

def tokenize_text(myData):  
  sequences = tokenizer.texts_to_sequences(myData)
  sequences_matrix_X = sequence.pad_sequences(sequences)
  return sequences_matrix_X

class Item(BaseModel):
    textRequest: str

def handleSingleClassify(path):
    data = pd.read_csv(path, header = None, sep = "/n")
    data.columns = ["text"]
    data = handle_preprocess_dataframe(data)
    X = data["processed_text"].astype(str)
    x_tokenized = tokenize_text(X)
    result_model = model.predict(x_tokenized)
    result = []
    for index, item in enumerate(result_model):
        if (item[0] > 0.45):
            result.append([data["text"][index], 1])
        else:
            result.append([data["text"][index], 0])
    return result

def handSingleText(text):
    print(text)
    text = handle_preprocess_single_text(text)
    print(text, 2)
    x_tokenized = tokenize_text([str(text)])
    print(x_tokenized)
    result_model = model.predict(x_tokenized)
    print(result_model[0][0])
    if result_model[0][0] > 0.45:
        return 1
    return 0


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/single")
async def handleSingle(item: Item):
    return handSingleText(item.textRequest)

@app.post("/file")
async def handleSingle(file: UploadFile  = File(...)):
    destination = "files/" + file.filename
    with open(destination, 'wb') as myFile:
        content = await file.read()
        myFile.write(content)
        myFile.close()
    return  handleSingleClassify(destination)