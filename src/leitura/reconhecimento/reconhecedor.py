import pytesseract

class Reconhecedor:
    def __init__(self, ficheiro_tesseract: str):
        pytesseract.pytesseract.tesseract_cmd = ficheiro_tesseract

    def reconhecer(self, imagem):
        texto = ""

        if imagem is not None:
            encontrado = pytesseract.image_to_string(imagem, config='--psm 11')

            # Eliminar tudo o que não seja alfanumérico
            texto += ''.join(filter(str.isalnum, encontrado))
        
        return texto
