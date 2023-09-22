import numpy as np
import cv2

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

if __name__ == '__main__':
    img = cv2.imread('dataset1/0a2de4c5-d688-4f9d-9107-ace1d281c307___Com.G_TgS_FL 7941_180deg.JPG')
    img_bi = cv2.imread('binarization_outputs/binarized_0.png')

    array_img = np.array(img)
    array_img_bi = np.array(img_bi)

    resultado = compute_fit_adjust(array_img_bi, array_img)

    print(resultado)