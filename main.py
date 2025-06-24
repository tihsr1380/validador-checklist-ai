from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import cv2

app = FastAPI()

# Carregar imagem de referência (modelo limpo)
modelo_path = "modelo_quarto_limpo.jpg"
modelo_ref = None
try:
    modelo_ref = cv2.imread(modelo_path)
    modelo_ref = cv2.resize(modelo_ref, (224, 224))
except:
    modelo_ref = None

def comparar_imagens(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(img1, img2)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = diff.size
    porcentagem_diferenca = (non_zero_count / total_pixels) * 100
    return porcentagem_diferenca

@app.post("/validar/")
async def validar_imagem(file: UploadFile = File(...)):
    if not modelo_ref is None:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img = np.array(image)
        img = cv2.resize(img, (224, 224))
        diff_percent = comparar_imagens(modelo_ref, img)
        if diff_percent < 15:
            return {"status": "aprovado", "diferenca": f"{diff_percent:.2f}%"}
        else:
            return {"status": "rejeitado", "diferenca": f"{diff_percent:.2f}%"}
    else:
        return JSONResponse(content={"erro": "Imagem modelo não carregada."}, status_code=500)
