import numpy as np
from tensorflow import keras

from src.config import num_classes, batch_size
from src.visualisation import show_original_image, show_original_annotation, show_predicted_annotation, show_combined_result


def get_model_prediction(trained_model, input_image):
    prediction = trained_model.predict(input_image)
    prediction = np.argmax(prediction, axis=-1)
    return prediction


def model_iou(trained_model, test_img_generator, test_anno_generator, show_images=True):
    metric = keras.metrics.MeanIoU(num_classes=num_classes)

    x = test_img_generator.next()
    y_true = test_anno_generator.next()
    y_pred = get_model_prediction(trained_model, x)

    metric.update_state(y_pred, y_true[:, :, :, 0])

    iou_score = np.round(metric.result().numpy(), decimals=2)

    if show_images:
        for i in range(batch_size):
            show_combined_result(model_input=x, y_true=y_true, y_pred=y_pred, i=i, iou_score=iou_score, save_file='results/combined1.png')
            # show_original_image(x, i=i)
            # show_original_annotation(y_true, i=i)
            # show_predicted_annotation(y_pred, i=i)

    return iou_score

