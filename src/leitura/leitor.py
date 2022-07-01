import cv2
import matplotlib.pyplot as plt

from leitura.detecao.detetor import Detetor
from leitura.reconhecimento.reconhecedor import Reconhecedor
from leitura.segmentacao.segmentador import Segmentador

class Leitor:
    def __init__(self, ficheiro_pesos: str, ficheiro_config: str, ficheiro_dnn: str, ficheiro_tesseract: str):
        self.__detetor      = Detetor(ficheiro_pesos, ficheiro_config)
        self.__segmentador  = Segmentador(ficheiro_dnn)
        self.__reconhecedor = Reconhecedor(ficheiro_tesseract)

    def processar(self, nome_ficheiro, debug=False):
        imagem = cv2.cvtColor(cv2.imread(nome_ficheiro), cv2.COLOR_BGR2RGB)

        matricula = self.__detetor.detetar(imagem)
        caracteres = self.__segmentador.segmentar(matricula)
        texto = self.__reconhecedor.reconhecer(caracteres)

        if debug and matricula is not None:
           
            print("Matrícula:", texto)
            plt.imshow(matricula)
            plt.show()
        
        return texto