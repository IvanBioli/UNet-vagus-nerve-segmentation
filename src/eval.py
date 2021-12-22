import numpy as np
from skimage.measure import label, regionprops
import cv2


def get_model_prediction(trained_model, img):
    """
        Predict the mask of an imput image

        Parametes
        ---------------
        trained_model: tf.keras.Model
            the trained model
        img: np.ndarray
            the input image to predict the mask

        Returns
        ---------------
        the predicted mask for the input image as a 1-channel numpy image
    """
    prediction = trained_model.predict(img)
    prediction = np.argmax(prediction, axis=-1)
    return prediction

def delete_small_regions(mask, threshold=101):
    """
        Remove regions from a mask which have area smaller than a threshold

        Parameters
        ---------------
        mask: np.ndarray
            the input mask
        threshold: int, optional
            the area (number of pixels) threshold

        Returns
        ---------------
        the mask after removing small regions
    """
    regions_mask = regionprops(label((mask * 255).astype(int)))
    regions_to_delete = [x for x in regions_mask if x.area < threshold]
    for x in regions_to_delete:
        for c in x.coords:            
            mask[c[0], c[1]] = 1 - mask[c[0], c[1]]
    return mask 

def apply_watershed(mask, coeff_list=[0.35]):
    """
        Apply the watershed algorithm to a mask

        Parameters
        ---------------
        mask: np.ndarray
            the input mask
        coeff_list: [float], optional
            list of thresholding coefficients for which the watershed algorithm is executed

        Returns
        ---------------
        the mask after applying the watershed algorithm
    """
    thresh = (mask * 255).astype('uint8')
    img = cv2.merge((thresh,thresh,thresh))
    
    # Morhphological operations to remove noise - morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
    # We find what we are sure is background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area using distance transform and thresholding
    # intensities of the points inside the foreground regions are changed to 
    # distance their respective distances from the closest 0 value (boundary).
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    for coeff in coeff_list:
        ret, sure_fg = cv2.threshold(dist_transform, coeff*dist_transform.max(),255,0)
        # Finding unknown region, the one that is not sure background and not sure foregound
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add 10 to all labels so that sure background is not 0, but 10
        markers = markers + 10
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        # Using watershed to have the markers
        markers = cv2.watershed(img, markers)
        # Dilatating watershed lines to have a better visualization
        watershed_lines = np.zeros(shape=np.shape(markers))
        watershed_lines[markers == -1] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        watershed_lines_thick = cv2.dilate(watershed_lines, kernel, iterations=1)
        # Drawing black watershed lines
        img[watershed_lines_thick == 1] = [0, 0, 0]

    # Converting back image to BG
    (_, img) = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    # Rescaling to [0, 1]
    img = img / 255

    return img

def predict_mask(trained_model, img, threshold = 0, coeff_list = None):
    """
        Predict the mask of an image and then apply pre-processing steps to the mask
        Pre-processing steps include: remove small regions and apply watershed

        Parameters
        ---------------
        trained_model: tf.keras.Model
            the trained model
        img: np.ndarray
            the input image to predict the mask
        threshold: int, optional
            threshold to apply delete_small_regions
        coeff_list: [float], optional
            coefficients for apply_watershed
    """
    prediction = get_model_prediction(trained_model, np.expand_dims(img, axis=0))[0, :, :]
    prediction = delete_small_regions(prediction, threshold)
    if coeff_list is not None:
        prediction = apply_watershed(prediction, coeff_list)
        prediction = delete_small_regions(prediction, threshold)
    return prediction