import cv2  # OpenCV
import numpy as np


imagem = cv2.imread('Flor_joaninha.jpg')
cv2.imshow('Imagem Original', imagem)


# Aplicando o filtro cvtColor para converter a imagem para escala de cinza
img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagem Cinza', img_cinza)


# Aplicando o filtro Canny para detecção de bordas na imagem
img_borda = cv2.Canny(img_cinza, 50, 150)
cv2.imshow('Imagem Borda', img_borda)


# Aplicando o filtro dilate e erode para melhorar a detecção de bordas e remoçao de ruidos, com duas interações para cada, a fim de reduzir os ruidos na imagem
img_dilatada = cv2.dilate(img_borda, None, iterations=2)
cv2.imshow('Imagem Dilatada', img_dilatada)
img_erosao = cv2.erode(img_dilatada, None, iterations=2)
cv2.imshow('Imagem Erosao', img_erosao)


# Encontrando o contorno da imagem a ser destacada na imagem original
contornos, hieraquia = cv2.findContours(img_erosao.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagem, contornos, -1, (0, 0, 0), 2)


# Aplicando a mascara para deixar a imagem original com os pixels todos em preto
mask = np.zeros_like(imagem)


# Em seguida foi utiizado o drawContours para desenhar os contornos da imagem a ser destacada
cv2.drawContours(mask, contornos, -1, (255,255,255), -1)
cv2.imshow('Imagem Mascara', mask)


# Resultado final da imagem após aplicação dos filtros e da mascara e com o bitwise_and para destacar a flor e a joaninha
resultado = cv2.bitwise_and(imagem, mask)
cv2.imshow('Imagem Resultado', resultado)


cv2.waitKey(0)
