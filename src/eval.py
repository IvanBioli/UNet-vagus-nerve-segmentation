import numpy as np
from tensorflow import keras
from skimage.measure import label, regionprops
import cv2
import matplotlib.pyplot as plt
from config import minimum_fascicle_area, watershed_coeff

from config import num_classes, batch_size


def get_model_prediction(trained_model, img):
    """
        Predict the mask of an imput image
        :param trained_model - the trained model
        :param img - the input image to predict the mask
    """
    prediction = trained_model.predict(img)
    prediction = np.argmax(prediction, axis=-1)
    return prediction

def delete_small_regions(mask, threshold=101):
    """
        Remove regions from a mask which have area smaller than a threshold
        :param mask - the input mask
        :param threshold - the area (number of pixels) threshold
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
        :param mask - the input mask
        :param coeff_list - TODO
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
        :param trained_model - the trained model
        :param img - the input image to predict the mask
        :param threshold - threshold to apply delete_small_regions
        :param coeff_list - coefficients for apply_watershed
    """
    prediction = get_model_prediction(trained_model, np.expand_dims(img, axis=0))[0, :, :]
    prediction = delete_small_regions(prediction, threshold)
    if coeff_list is not None:
        prediction = apply_watershed(prediction, coeff_list)
        prediction = delete_small_regions(prediction, threshold)
    return prediction


'''
def one_prediction_iou(trained_model, test_img, test_anno, show_images=True):
    metric = keras.metrics.MeanIoU(num_classes=num_classes)

    x = np.expand_dims(test_img, axis=0)
    y_true = test_anno
    y_pred = get_model_prediction(trained_model, x)

    metric.update_state(y_pred, y_true)

    iou_score = np.round(metric.result().numpy(), decimals=2)

    if show_images:
        for i in range(batch_size):
            # show_combined_result(model_input=x, y_true=y_true, y_pred=y_pred, i=i, iou_score=iou_score, save_file='results/combined1.png')
            # show_overlay_result(model_input=x, y_true=y_true, y_pred=y_pred, i=i, iou_score=iou_score, save_file='results/overlay2.png')
            show_result_test(x[0, :, :, :], y_pred[0, :, :])
            # show_original_image(x, i=i)
            # show_original_annotation(y_true, i=i)
            # show_predicted_annotation(y_pred, i=i)

    return iou_score


def model_iou(trained_model, test_img_generator, test_anno_generator, show_images=True):
    metric = keras.metrics.MeanIoU(num_classes=num_classes)

    x = test_img_generator.next()
    y_true = test_anno_generator.next()
    y_pred = get_model_prediction(trained_model, x)

    metric.update_state(y_pred, y_true[:, :, :, 0])

    iou_score = np.round(metric.result().numpy(), decimals=2)

    if show_images:
        for i in range(batch_size):
            # show_combined_result(model_input=x, y_true=y_true, y_pred=y_pred, i=i, iou_score=iou_score, save_file='results/combined1.png')
            show_overlay_result(model_input=x, y_true=y_true, y_pred=y_pred, i=i, iou_score=iou_score,
                                save_file='results/overlay2.png')
            # show_original_image(x, i=i)
            # show_original_annotation(y_true, i=i)
            # show_predicted_annotation(y_pred, i=i)

    return iou_score
'''
