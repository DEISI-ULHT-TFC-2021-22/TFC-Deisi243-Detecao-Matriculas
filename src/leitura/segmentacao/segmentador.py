import cv2
import numpy as np
import matplotlib.pyplot as plt

from imutils.object_detection import non_max_suppression

class Segmentador:
    def __init__(self, ficheiro_dnn: str):
        self.__dnn = cv2.dnn.readNet(ficheiro_dnn)

    def segmentar(self, imagem):
        if imagem is None:
            return

        # Guardar imagem original
        original = imagem.copy()

        # Converter para grayscale
        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

        # Gaussian Blur
        imagem = cv2.GaussianBlur(imagem, (3,3), 0)

        # Método de Otsu
        _, imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = np.ones((3,3),np.uint8)
        imagem = cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, kernel)
        imagem = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, kernel)

        imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2RGB)

        # Dimensões múltiplas de 32
        h, w = np.clip(imagem.shape[0] // 32, 1, None) * 32, np.clip(imagem.shape[1] // 32, 1, None) * 32
        rh, rw = original.shape[0]/h, original.shape[1]/w

        # Redimensionar a imagem
        imagem = cv2.resize(imagem, (w,h))

        blob = cv2.dnn.blobFromImage(imagem, 1.0, (w,h), (123.68, 116.78, 103.94), swapRB=True, crop=False)

        outputLayers = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ]

        self.__dnn.setInput(blob)
        output = self.__dnn.forward(outputLayers)
        scores = output[0]
        geometry = output[1]

        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < 0.5:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        rect = [[original.shape[1],original.shape[0]], [0,0]]

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(np.clip(startX * rw - 5, 0, original.shape[1]))
            startY = int(np.clip(startY * rh - 5, 0, original.shape[0]))
            endX = int(np.clip(endX * rw + 5, 0, original.shape[1]))
            endY = int(np.clip(endY * rh + 5, 0, original.shape[0]))
            
            if startX < rect[0][0]:
                rect[0][0] = startX
            if startY < rect[0][1]:
                rect[0][1] = startY
            if endX > rect[1][0]:
                rect[1][0] = endX
            if endY > rect[1][1]:
                rect[1][1] = endY

        if rect[0][1] <= rect[1][1] and rect[0][0] <= rect[1][0]:
            return original[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]