import numpy as np
import cv2
import os
import json

def compute_fit_adjust(array, arrayRef):
    """
    This functions computes the Fit Adjust and returns the computed value
    according to the formula:

    Arguments:
        array {numpy.array} -- input array with segmented content
        array {numpy.array} -- input array with reference content
        
    Returns: 
        float -- Fit Adjust value between 0 and 1.
    """
    imand = np.bitwise_and(array.astype("uint8"), arrayRef.astype("uint8"))
    imor = np.bitwise_or(array.astype("uint8"), arrayRef.astype("uint8"))
    sumand = np.sum(imand)
    sumor = np.sum(imor)

    result = (sumand / float(sumor))

    return result

def compute_size_adjust(array, arrayRef):
    """
    This functions computes the Size Adjust and returns the computed value:

    Arguments:
        array {numpy.array} -- input array with segmented content
        array {numpy.array} -- input array with reference content
        
    Returns: 
        float -- Fit Adjust value between 0 and 1.
    """
    imArea1 = np.count_nonzero(arrayRef)
    imArea2 = np.count_nonzero(array)
    subArea = np.abs(imArea1 - imArea2)
    sumArea = imArea1 + imArea2

    result = (1 - subArea / sumArea)

    return result

def compute_position_adjust(arraySeg, arrayRef):
    """
    This functions computes the Size Adjust and returns the computed value:

    Arguments:
        array {numpy.array} -- input array with segmented content
        array {numpy.array} -- input array with reference content
        
    Returns: 
        float -- Fit Adjust value between 0 and 1.
    """
    indsSeg = np.where(arraySeg > 0)
    indsRef = np.where(arrayRef > 0)

    centroidRefY = indsRef[0].mean()
    centroidRefX = indsRef[1].mean()

    centroidSegY = indsSeg[0].mean()
    centroidSegX = indsSeg[1].mean()

    subCentroidY = np.abs(centroidSegY - centroidRefY) / arrayRef.shape[0]
    subCentroidX = np.abs(centroidSegX - centroidRefX) / arrayRef.shape[1]

    result = 1 - (subCentroidY + subCentroidX) / 3

    return result

def compute_dice_similarity(array, arrayRef):
    """
    This functions computes the Dice Similarity Coefficient and returns the computed value
    according to the formula:

    .. math::
    DSC = 2 * (A_seg ∩ A_ref) (|A_seg| + |A_ref|)

    Arguments:
        array {numpy.array} -- input array with segmented content
        array {numpy.array} -- input array with reference content
        
    Returns: 
        float -- Fit Adjust value between 0 and 1.
    """
    imand = np.bitwise_and(array.astype("uint8"), arrayRef.astype("uint8"))
    imor = np.bitwise_or(array.astype("uint8"), arrayRef.astype("uint8"))
    sumand = 2 * np.sum(imand)
    sumor = np.sum(array) + np.sum(arrayRef)

    result = (sumand / float(sumor))

    return result

def calcular_metricas_agregadas(fit_valores, size_valores, position_valores, dice_valores):

    # Calcule os valores agregados
    minimo = {
        'Dice': min(dice_valores),
        'Fit': min(fit_valores),
        'Size': min(size_valores),
        'Position': min(position_valores)
    }

    maximo = {
        'Dice': max(dice_valores),
        'Fit': max(fit_valores),
        'Size': max(size_valores),
        'Position': max(position_valores)
    }

    media = {
        'Dice': np.mean(dice_valores),
        'Fit': np.mean(fit_valores),
        'Size': np.mean(size_valores),
        'Position': np.mean(position_valores)
    }

    desvio_padrao = {
        'Dice': np.std(dice_valores),
        'Fit': np.std(fit_valores),
        'Size': np.std(size_valores),
        'Position': np.std(position_valores)
    }

    return minimo, maximo, media, desvio_padrao

if __name__ == '__main__':

    # Pastas onde as imagens estão localizadas
    pasta_img_manuais = 'd:\Coisas\Datasets\Folhas\manual_seg\Potato___healthy'
    pasta_img_automaticas = 'd:\Coisas\Datasets\Folhas\imgs_segmentadas\Potato_healthy'

    # Lista de extensões de arquivo de imagem suportadas
    extensoes_suportadas = ['.jpg', '.jpeg', '.png']

    # Listar os arquivos nas duas pastas simultaneamente
    arquivos_pasta_manuais = os.listdir(pasta_img_manuais)
    arquivos_pasta_automaticas = os.listdir(pasta_img_automaticas)

    # inicia os vetores
    dice_valores = []
    fit_valores = []
    size_valores = []
    position_valores = []

    # Certifique-se de que as duas pastas tenham o mesmo número de imagens
    if len(arquivos_pasta_manuais) != len(arquivos_pasta_automaticas):
        print("As pastas não têm o mesmo número de imagens.")
    else:
        for arquivos_pasta_manuais, arquivos_pasta_automaticas in zip(arquivos_pasta_manuais, arquivos_pasta_automaticas):
            if any(arquivos_pasta_manuais.lower().endswith(ext) and arquivos_pasta_automaticas.lower().endswith(ext) for ext in extensoes_suportadas):
                caminho_imagem_manual = os.path.join(pasta_img_manuais, arquivos_pasta_manuais)
                caminho_imagem_automatica = os.path.join(pasta_img_automaticas, arquivos_pasta_automaticas)

                img_manual = cv2.imread(caminho_imagem_manual)
                img_automatica = cv2.imread(caminho_imagem_automatica)

                if img_manual is not None and img_automatica is not None:
                    array_img_manual = np.array(img_manual)
                    array_img_automatica = np.array(img_automatica)

                    resultado_fit = compute_fit_adjust(array_img_automatica, array_img_manual)
                    resultado_size = compute_size_adjust(array_img_automatica, array_img_manual)
                    resultado_position = compute_position_adjust(array_img_automatica, array_img_manual)
                    resultado_dice = compute_dice_similarity(array_img_automatica, array_img_manual)

                    dice_valores.append(resultado_fit)
                    fit_valores.append(resultado_size)
                    size_valores.append(resultado_position)
                    position_valores.append(resultado_dice)
                    
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print(f"Erro ao carregar as imagens {arquivos_pasta_manuais} e {arquivos_pasta_automaticas}")
    
    retorno = calcular_metricas_agregadas(fit_valores, size_valores, position_valores, dice_valores)
    nome_do_grupo = 'potato_healthy'
    
    print(retorno)

    resultados = {
                'Nome do Arquivo': nome_do_grupo,
                'Min': retorno[0],
                'Max': retorno[1],
                'Media': retorno[2],
                'Desvio Padrao': retorno[3]
            }
    
    nome_json = 'resultados.json'

    with open(nome_json, 'a') as arquivo_json:
        # Verifica se o arquivo já possui conteúdo
        if os.stat(nome_json).st_size == 0:
            # Se o arquivo estiver vazio, escreve o início do JSON
            arquivo_json.write('[\n')
        else:
            # Se o arquivo já possui conteúdo, posiciona o cursor no final
            arquivo_json.seek(0, 2)
            arquivo_json.write(',\n')

        # Escreve os resultados no arquivo
        json.dump(resultados, arquivo_json, indent=4)

    print("Resultados adicionados em 'resultados.json'")

    #=================================================================

    #  img_manual = cv2.imread('d:/Coisas/Datasets/Folhas/manual_seg/Strawberry___healthy/00e9a277-ca5e-4350-95ce-8b2918b69fb9___RS_HL 4667.png')
    # img_automatico = cv2.imread('d:/Coisas/Datasets/Folhas/imgs_segmentadas/Strawberry__healthy/binarized_0.png')

    # array_img = np.array(img_manual)
    # array_img_bi = np.array(img_automatico)

    # resultado = compute_fit_adjust(array_img_bi, array_img)

    # print(resultado)