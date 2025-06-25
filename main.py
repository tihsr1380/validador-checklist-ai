from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from starlette.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Permitir requisições de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/validate")
async def validate_image(file: UploadFile = File(...)):
    try:
        # Verifica se é imagem
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Arquivo enviado não é uma imagem.")

        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        uploaded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if uploaded_img is None:
            raise HTTPException(status_code=400, detail="Erro ao ler a imagem enviada.")

        # Caminho absoluto da imagem modelo
        modelo_path = os.path.join(os.path.dirname(__file__), "modelo.jpg")
        if not os.path.exists(modelo_path):
            raise HTTPException(status_code=500, detail="Imagem modelo não encontrada no servidor.")

        modelo_img = cv2.imread(modelo_path)
        if modelo_img is None:
            raise HTTPException(status_code=500, detail="Erro ao carregar imagem modelo.")

        # Redimensiona ambas para comparação justa
        resized_uploaded = cv2.resize(uploaded_img, (300, 300))
        resized_modelo = cv2.resize(modelo_img, (300, 300))

        # Compara diferença absoluta
        diff = cv2.absdiff(resized_uploaded, resized_modelo)
        mse = np.mean(diff ** 2)

        LIMIAR = 500  # ajuste esse valor se necessário
        conforme = mse < LIMIAR

        return JSONResponse(content={"conforme": conforme, "mse": float(mse)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
