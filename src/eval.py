import numpy as np
from tensorflow import keras
from skimage.measure import label, regionprops
import cv2
import matplotlib.pyplot as plt

from src.config import num_classes, batch_size
from src.visualisation import show_overlay_result, show_result_test
#from config import num_classes, batch_size
#from visualisation import show_overlay_result, show_result_test


def get_model_prediction(trained_model, input_image):
    prediction = trained_model.predict(input_image)
    prediction = np.argmax(prediction, axis=-1)
    return prediction

def delete_small_regions(input_mask, threshold = 40):
    regions_mask = regionprops(label(((1 - input_mask) * 255).astype(int)))
    regions_to_delete = [x for x in regions_mask if x.area < threshold]
    for x in regions_to_delete:
        for c in x.coords:            
            input_mask[c[0], c[1]] = 1 - input_mask[c[0], c[1]]
    return input_mask 

def apply_watershed(mask, coeff_list=[0.35]):
    thresh = ((1 - mask) * 255).astype('uint8')
    img = cv2.merge((thresh,thresh,thresh))
    
    # Morhphological operations to remove noise - morphological opening
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # We find what we are sure is background
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area using distance transform and thresholding
    # intensities of the points inside the foreground regions are changed to 
    # distance their respective distances from the closest 0 value (boundary).
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    for i, coeff in enumerate(coeff_list):
        ret, sure_fg = cv2.threshold(dist_transform, coeff*dist_transform.max(),255,0)
        # Finding unknown region, the one that is not sure background and not sure foregound
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add 10 to all labels so that sure background is not 0, but 10
        markers = markers + 10
        # Mark the region of unknown with zero
        markers[unknown==255] = 0
        # Using watershed to have the markers
        markers = cv2.watershed(img,markers)
        # We draw a black border according to the markers
        img[markers == -1] = [0,0,0]

    (_, mask_wat) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    assert((np.unique(mask_wat) == [0, 255]).all())
    mask_wat = 1 - mask_wat / 255 
    
    return mask_wat

def predict_mask(trained_model, img, threshold = 15, coeff_list = [0.1, 0.35, 0.37, 0.4]):
    prediction = get_model_prediction(trained_model, np.expand_dims(img, axis=0))[0, :, :]
    prediction = delete_small_regions(prediction, threshold)
    prediction = apply_watershed(prediction, coeff_list)
    # To delete artifacts of the watershed
    prediction = delete_small_regions(prediction, threshold)
    prediction[prediction == 1.] = 1
    prediction[prediction == 0.] = 0
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
