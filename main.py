from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

app = FastAPI()

# Libera CORS para seu sistema PHP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restrinja ao seu domínio
    allow_methods=["*"],
    allow_headers=["*"],
)

def comparar_imagens(imagem1, imagem2):
    # Redimensiona para o mesmo tamanho
    imagem1 = cv2.resize(imagem1, (400, 400))
    imagem2 = cv2.resize(imagem2, (400, 400))

    # Converte para escala de cinza
    cinza1 = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
    cinza2 = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)

    # Calcula histograma e normaliza
    hist1 = cv2.calcHist([cinza1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([cinza2], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Calcula correlação (quanto mais perto de 1, mais similar)
    similaridade = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # ⚠️ LOG de debug:
    print(f"[DEBUG] Similaridade entre imagem enviada e modelo: {similaridade:.4f}")

    # Considera conforme se similaridade for >= 0.80
    conforme = similaridade >= 0.80
    return conforme

@app.post("/validar-imagem/")
async def validar_imagem(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    imagem_recebida = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    modelo = cv2.imread("modelo_quarto_limpo.jpg")
    if modelo is None:
        return {"erro": "Imagem modelo não encontrada no servidor."}

    resultado = comparar_imagens(imagem_recebida, modelo)
    return {"conforme": resultado}
