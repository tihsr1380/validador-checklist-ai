from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np

app = FastAPI()

@app.get("/")
def root():
    return {"status": "online"}

@app.post("/validate")
async def validate_image(file: UploadFile = File(...)):
    try:
        # Lê a imagem enviada
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_recebida = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_recebida is None:
            return JSONResponse(content={"erro": "Imagem inválida"}, status_code=400)

        # Lê a imagem modelo salva no projeto com nome 'modelo.jpg'
        modelo = cv2.imread("modelo.jpg")

        if modelo is None:
            return JSONResponse(content={"erro": "Imagem modelo 'modelo.jpg' não encontrada"}, status_code=500)

        # Redimensiona a imagem recebida para o tamanho do modelo
        img_recebida = cv2.resize(img_recebida, (modelo.shape[1], modelo.shape[0]))

        # Calcula a diferença entre as imagens
        diff = cv2.absdiff(modelo, img_recebida)
        mean_diff = np.mean(diff)

        LIMITE_TOLERANCIA = 20  # Valor de referência ajustável

        if mean_diff > LIMITE_TOLERANCIA:
            return {
                "status": "fora_do_padrao",
                "mensagem": "A imagem está fora do padrão do ambiente modelo.",
                "diferenca": float(mean_diff)
            }
        else:
            return {
                "status": "ok",
                "mensagem": "A imagem está dentro do padrão esperado.",
                "diferenca": float(mean_diff)
            }

    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)
