import glob
from leitura.leitor import Leitor

if __name__ == "__main__":
    leitor = Leitor(
        "./config/yolo/lapi.weights",
        "./config/yolo/darknet-yolov3.cfg",
        "./config/east/frozen_east_text_detection.pb",
        "C:\\Users\\Diogo Pereira\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe")

    for ficheiro in glob.glob("C:\\Users\\Diogo Pereira\\Desktop\\TFC\\dataset\\*"):
        texto = leitor.processar(ficheiro, debug=True)