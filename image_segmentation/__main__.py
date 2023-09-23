import cv2
import os
from functions import *

if __name__ == '__main__':
    # img = cv2.imread('dataset/0a2de4c5-d688-4f9d-9107-ace1d281c307___Com.G_TgS_FL 7941_180deg.JPG')
    
    # delete_outputs('d:\Coisas\Datasets\Folhas\imgs_segmentadas\Blueberry__healthy') #esvazia a pasta output

    pasta_imgs = 'd:\Coisas\Datasets\Folhas\img_para_segmentacao\Blueberry__healthy'
    arquivos = os.listdir(pasta_imgs)  
    contador = 0
    #loop que pega os arquivos na pasta dataset
    for arq in arquivos:
        #condição que checa se o arquivo termina com a extensão .jpg
        if arq.lower().endswith('jpg'):
            caminho_img = os.path.join(pasta_imgs, arq)
            img = cv2.imread(caminho_img)

            #condição que checa se img não está vazio
            if img is not None:           
                #recebe a função de checagem de contraste
                img_contrast_check = contrast_check(img)

                #aplicar borramento gaussiano para reduzir ruídos
                borrado_mask = cv2.GaussianBlur(img_contrast_check, (3, 3), 0) #mais pixels = menos ruído e mais imperfeições!
                
                hsv = cv2.cvtColor(borrado_mask, cv2.COLOR_BGR2HSV)
                
                mask = create_mask(hsv)
                
                mask_fechamento = fechamento(mask)  

                mask_contornos = find_draw_contours(mask_fechamento)
                
                #separação do fundo de imagem com a região de interesse
                imagem_roi = background_separation(img_contrast_check, img_contrast_check, mask_contornos)

                imagem_lbp = structuring_lbp(imagem_roi)

                cv2.imshow('Imagem original', img)
                cv2.imshow('Máscara da matiz', mask)
                cv2.imshow('Fechamento', mask_fechamento)
                cv2.imshow('Maior Contorno', mask_contornos)
                cv2.imshow('No backgorund', imagem_roi)
                cv2.imshow('Binarization', imagem_lbp)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # comentem esse trecho para não gerar arquivos
                output_caminho = naming_outputs(contador) # esse é a função que gera o caminho, será necessário modifica-lá para gerar um caminho próprio
                cv2.imwrite(output_caminho, mask_contornos)
            else:
                print(f'Erro ao carregar imagem {arq}')
        
        contador += 1