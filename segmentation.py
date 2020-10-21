#Bibliotecas
import cv2
import numpy as np
import matplotlib.pyplot as plt

def getCircle(n):
    '''kernel has size NxN'''
    # xx and yy are 200x200 tables containing the x and y coordinates as values
    # mgrid is a mesh creation helper
    xx, yy = np.mgrid[:n,:n]
    # circles contains the squared distance to the (100, 100) point
    # we are just using the circle equation learnt at school
    circle = (xx - np.floor(n/2)) ** 2 + (yy - np.floor(n/2)) ** 2
    circle = circle<=np.max(circle)*.5
    circle = np.uint8(circle)
    return circle

#Função de processamento principal
def processing(img, parameters):
    '''
    Função de processamento principal.

    A entrada é uma imagem BGR (resultado da função cv2.imread(), por exemplo); e uma lista de parâmetros opcional
    Cada posição da lista de parâmetros corresponde a:
        0 thresholdType = cv2.ADAPTIVE_THRESH_MEAN_C or cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        1 blockSize = 99-299
        2 constant = 0-20
        3 kernelSize = [3-7]
        4 openingIt = 0-2
        5 erosionIt = 0-5
        6 contourMethod = cv2.CHAIN_APPROX_SIMPLE or cv2.CHAIN_APPROX_TC89_L1 or cv2.CHAIN_APPROX_TC89_KCOS 
        7 minArea = 15-50
    
    A saída é uma tupla de 5 elementos:
        img_contours = contornos em uma cor desenhados por cima da imagem original
        img_borders  = contornos coloridos destacando os grãos das bordas
        img_colored  = contornos coloridos aleatoriamente
        img_out      = contornos coloridos aleatoriamente desenhados por cima da imagem original
        resultado    = int representando o número de grãos contados
    '''
    #Default parameters
    if parameters == None:
        #parameters = [0,199,3,3,0,3,0,20]
        parameters = [1, 191, 3, 5, 0, 2, 0, 41]
    
    #Transformação do vetor de parâmetros em variáveis com nomes informativos
    if parameters[0] == 0: thresholdType = cv2.ADAPTIVE_THRESH_MEAN_C
    if parameters[0] == 1: thresholdType = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    blockSize =  parameters[1]
    constant =   parameters[2]
    kernelSize = parameters[3]
    openingIt =  parameters[4]
    erosionIt =  parameters[5]
    if parameters[6] == 0: contourMethod = cv2.CHAIN_APPROX_SIMPLE
    if parameters[6] == 1: contourMethod = cv2.CHAIN_APPROX_TC89_L1
    if parameters[6] == 2: contourMethod = cv2.CHAIN_APPROX_TC89_KCOS
    minArea =    parameters[7]
    
    #Conversão de uma imagem para outro sistema de cores
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Adaptivo
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, thresholdType, cv2.THRESH_BINARY, blockSize, constant) 

    #Fechamento   
    kernel = getCircle(kernelSize)
    img_open = cv2.morphologyEx(img_thresh,cv2.MORPH_CLOSE,kernel, iterations = openingIt)
    img_open = cv2.erode(img_open, kernel, iterations=erosionIt)

    #Desenhando borda na imagem
    y,x = img_open.shape
    color = 0
    img_open[:,   0] = 0; img_open[:, x-1] = 0; img_open[0,   :] = 0; img_open[y-1, :] = 0

    #Gerando Lista de Contornos
    cv2MajorVersion = cv2.__version__.split(".")[0]
    if int(cv2MajorVersion) >= 4:
        contours, _= cv2.findContours(img_open,cv2.RETR_EXTERNAL, contourMethod)
    else:
        _, contours, _ = cv2.findContours(img_open,cv2.RETR_EXTERNAL, contourMethod)

    #Ordenando Lista de Contornos de acordo com a área
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    #Selecionando apenas contornos cuja área é maior que algum valor
    contours = [c for c in contours if cv2.contourArea(c)>minArea]

    #Desenhando Contornos na imagem original
    verde = (0,255,0)
    img_contours = cv2.drawContours(img.copy(), contours, -1, verde, 3)

    #Separando grãos das bordas
    faixa = 3
    n_borda = 0
    img_borders = np.int32(np.ones(img.shape))
    red = [0,255,0]
    blue = [255,0,0]
    for c in contours:
        (x_ini,y_ini,w,h) = cv2.boundingRect(c)
        x_end = x_ini+w; y_end = y_ini+h
        y_img, x_img = img_thresh.shape

        if 0<x_ini<faixa or 0<y_ini<faixa or x_img-faixa<x_end<x_img or y_img-faixa<y_end<y_img:
            n_borda +=1
            random_red = [np.random.randint(20, 235) for i in range(3)]
            random_red[0] = 255
            img_borders = cv2.fillPoly(img_borders, [c], random_red)
        else:
            random_blue = [np.random.randint(20, 235) for i in range(3)]
            random_blue[1] = 255
            img_borders = cv2.fillPoly(img_borders, [c], random_blue)

    #Preenchendo contornos
    img_colored = np.int32(np.ones(img.shape))
    img_out = img.copy()
    for c in contours:
        random_color = [np.random.randint(20, 235) for i in range(3)]
        img_colored = cv2.fillPoly(img_colored, [c], random_color)
        img_out = cv2.drawContours(img_out, [c], -1, random_color, 3)
    
    #resultados
    resultado = len(contours)-round(n_borda/2)
    
    return img_contours, img_borders, img_colored, img_out, resultado

if __name__ == "__main__":
    img = cv2.imread("data/Alu b.jpg")
    parametros = [1, 191, 3, 5, 0, 2, 0, 41]
    img_contours, img_borders, img_colored, img_out, resultado = processing(img, parametros)

    print("Resultado = {} grãos contados".format(resultado))

    #plt.figure(figsize=(10,10))
    #plt.axis('off')
    #plt.imshow(img_out, 'gray')
    #plt.savefig('output.png')

    cv2.imwrite('output.jpg',img_out)

