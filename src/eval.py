import numpy as np
from tensorflow import keras

# from src.config import num_classes, batch_size
# from src.visualisation import show_overlay_result, show_result_test
from config import num_classes, batch_size
from visualisation import show_overlay_result, show_result_test


def get_model_prediction(trained_model, input_image):
    prediction = trained_model.predict(input_image)
    prediction = np.argmax(prediction, axis=-1)
    return prediction


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
            show_overlay_result(model_input=x, y_true=y_true, y_pred=y_pred, i=i, iou_score=iou_score, save_file='results/overlay2.png')
            # show_original_image(x, i=i)
            # show_original_annotation(y_true, i=i)
            # show_predicted_annotation(y_pred, i=i)

    return iou_score
