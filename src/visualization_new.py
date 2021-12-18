import os
import pickle
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorflow import keras
import tensorflow_addons as tfa
from loss import dice_loss, nerve_segmentation_loss, tversky_loss, iou_score, focal_tversky_loss, focal_loss, custom_loss
from eval import predict_mask, get_model_prediction
from stats import get_samples, calculate_regions, compute_bins
from augmentation import get_random_affine_transformation

from config import initialise_run, model_path, minimum_fascicle_area, watershed_coeff

custom = {'iou_score': iou_score, 'dice_loss': dice_loss, 'nerve_segmentation_loss': nerve_segmentation_loss, 'tversky_loss': tversky_loss, 'focal_tversky_loss': focal_tversky_loss, 'SigmoidFocalCrossEntropy': tfa.losses.SigmoidFocalCrossEntropy(), 'focal_loss': focal_loss, 'custom_loss': custom_loss}

def plot_augmented_images(img_path, num_aug=6, num_aug_wcolor=2, save=False, show=True):

    img = np.load(img_path)

    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/augmentations')
        os.makedirs(output_folder, exist_ok=True)
        out_fname = output_folder

    augmented_imgs = []
    augmented_imgs_wcolor = []

    for _ in range(num_aug):
        transform = get_random_affine_transformation()
        augmented_imgs.append(transform(img, do_colour_transform=False))
    
    for _ in range(num_aug_wcolor):
        transform = get_random_affine_transformation()
        augmented_imgs_wcolor.append(transform(img))

    # augmented_imgs = np.array(augmented_imgs)
    # augmented_imgs_wcolor = np.array(augmented_imgs_wcolor)

    augmented_imgs= np.reshape(augmented_imgs, (2, num_aug // 2, 512, 512, 3))
    augmented_imgs_wcolor = np.reshape(augmented_imgs_wcolor, (2, num_aug_wcolor // 2, 512, 512, 3))
    
    num_col = (num_aug + num_aug_wcolor) // 2 + 2
    fig, axs = plt.subplots(2, num_col, figsize=(num_col * 3, 6))

    gs = axs[0][0].get_gridspec()

    for row in range(2): 
        for ax in axs[row][0:2]:
            ax.remove()
    
    original_ax = fig.add_subplot(gs[0:2, 0:2])
    original_ax.imshow(img)

    for x in range(2):
        for y in range(2, num_aug // 2 + 2):
            axs[x][y].imshow(augmented_imgs[x][y - 2])
            axs[x][y].axis('off')

        for y_wcolor in range(-num_aug_wcolor // 2, 0):
            axs[x][y_wcolor].imshow(augmented_imgs_wcolor[x][y_wcolor + num_aug_wcolor // 2])
            axs[x][y_wcolor].axis('off')

    plt.axis('off')

    if save:
        plt.savefig(out_fname + '/augmentations_visualization.png')
    if show:
        plt.show()
        

def plot_masks_vs_predictions(path_list, trained_model_checkpoint=None, wstats=False, save=False, show=True):
    if trained_model_checkpoint is not None:
        trained_model = keras.models.load_model(trained_model_checkpoint, custom_objects=custom)

    if wstats:
        sub_folder = 'wstats'
    else:
        sub_folder = 'default'

    fig, axs = plt.subplots(len(path_list), 4, figsize=(4 * 2, len(path_list) * 2))

    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/predictions/', sub_folder)
        os.makedirs(output_folder, exist_ok=True)
        out_fname = output_folder

    for k, path in enumerate(path_list):
        img = np.load(path[0])
        mask = np.load(path[1])

        pred = predict_mask(trained_model, img)

        axs[k, 0].imshow(img)
        axs[k, 1].imshow(mask, cmap='gray', interpolation='none')
        axs[k, 2].imshow(pred, cmap='gray', interpolation='none')

        axs[k, 3].imshow(mask, cmap='gray', interpolation='none')
        axs[k, 3].imshow(pred, cmap='viridis', alpha=0.5, interpolation='none')

        if wstats:
            iou = str(np.around(iou_score(mask, pred, logits=False).numpy(), decimals=3))
            axs[k, 3].set_xlabel('IoU = ' + iou)
        for i in range(4):
            axs[k, i].xaxis.set_major_locator(ticker.NullLocator())
            axs[k, i].yaxis.set_major_locator(ticker.NullLocator())

    axs[0, 0].set_title('Input image')
    axs[0, 1].set_title('Ground truth')
    axs[0, 2].set_title('Prediction')
    axs[0, 3].set_title('Prediction overlayed\n on ground truth')

    plt.tight_layout()

    if save:
        plt.savefig(out_fname + '/sample_predictions_' + sub_folder + '.png')
    if show:
        plt.show()

def plot_fascicles_distribution(paths, test=False, trained_model_checkpoint=None, save=False, show=True, postprocessing = False):
    
    if trained_model_checkpoint is not None:
        trained_model = keras.models.load_model(trained_model_checkpoint, custom_objects = custom)

    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/distributions')
        os.makedirs(output_folder, exist_ok=True)
        if test:
            fname = 'distribution_unlabelled_test_set'
        else:
            fname = 'distribution_training_set'
        out_fname = os.path.join(output_folder, fname)

    if not test:
        areas_mask = []
        num_fascicles_mask = []
        eccentricity_mask = []
    areas_pred = []
    num_fascicles_pred = []
    eccentricity_pred = []

    for p in paths:
        if not test:
            img_path, mask_path = p
            mask = np.load(mask_path)
        else:
            img_path = p
            mask = None

        img = np.load(img_path)
        if postprocessing:
            pred = predict_mask(trained_model, img, threshold=minimum_fascicle_area, coeff_list=watershed_coeff)
        else:
            pred = predict_mask(trained_model, img)

        regions_pred, regions_mask = calculate_regions(pred, mask)
        
        pred_post = predict_mask(trained_model, img, threshold=minimum_fascicle_area)
        regions_post, _ = calculate_regions(pred_post)
            
        if not test:
            areas_mask = areas_mask + [m.area for m in regions_mask]
            eccentricity_mask = eccentricity_mask + [m.eccentricity for m in regions_mask]
            num_fascicles_mask.append(len(regions_mask))
        areas_pred = areas_pred + [p.area for p in regions_pred]
        eccentricity_pred = eccentricity_pred + [p.eccentricity for p in regions_pred]
        num_fascicles_pred.append(len(regions_pred))


    fig, axs = plt.subplots(1, 3, figsize=(10, 6))

    nbins_area = 50
    nbins_fascicles = 10
    nbins_eccentricity = 50
    if not test:
        # Computing the bins for the areas' histogram and plotting the histogram
        bins_areas = compute_bins(nbins_area, areas_pred, areas_mask)
        axs[0].hist(areas_mask, bins=bins_areas, alpha=0.5, label='Ground truth')
        # Computing the bins for the fascicles' histogram and plotting the histogram
        bins_fascicles = compute_bins(nbins_fascicles, num_fascicles_pred, num_fascicles_mask)
        axs[1].hist(num_fascicles_mask, bins=bins_fascicles, alpha=0.5, label='Ground truth')
        bins_eccentricity = compute_bins(nbins_eccentricity, eccentricity_pred, eccentricity_mask)
        axs[2].hist(eccentricity_mask, bins=bins_eccentricity, alpha=0.5, label='Ground truth')
    else:
        bins_areas = compute_bins(nbins_area, areas_pred)
        bins_fascicles = compute_bins(nbins_fascicles, num_fascicles_pred)
        bins_eccentricity = compute_bins(nbins_eccentricity, eccentricity_pred)

    # Histogram of Fascicles Areas for the predictions
    axs[0].hist(areas_pred, bins=bins_areas, alpha=0.5, label='Prediction')
    axs[0].set_xlabel('Areas (pixels)')
    axs[0].set_ylabel('Occurrencies')
    axs[0].legend(loc='upper right')
    axs[0].set_title('Histogram of Fascicles Areas')

    # Histogram of Number of fascicles Areas for the predictions
    axs[1].hist(num_fascicles_pred, bins=bins_fascicles, alpha=0.5, label='Prediction')
    axs[1].set_xlabel('Number of Fascicles')
    axs[1].set_ylabel('Occurrencies')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Histogram of Number of Fascicles')

    # Histogram of the Eccentricity
    axs[2].hist(eccentricity_pred, bins=bins_eccentricity, alpha=0.5, label='Prediction')
    axs[2].set_xlabel('Eccentricity')
    axs[2].set_ylabel('Occurrencies')
    axs[2].legend(loc='upper right')
    axs[2].set_title('Histogram of Eccentricity')

    # Print the quantiles of the areas and the eccentricity for the masks:
    if not test:
        print('0.01-quantile of the masks'' area:', np.quantile(areas_mask, 0.01))
        print('0.99-quantile of the masks'' area:', np.quantile(areas_mask, 0.99))
        print('0.01-quantile of the masks'' eccentricity:', np.quantile(eccentricity_mask, 0.01))
        print('0.99-quantile of the masks'' eccentricity:', np.quantile(eccentricity_mask, 0.99))

    # If postprocessing is involved
    if save:
        if postprocessing:
            out_fname = out_fname + '_postprocessed'
        plt.savefig(out_fname + '.png')
    if show:
        plt.show()

def plot_postprocessed(path_list, trained_model_checkpoint=None, save=False, show=True, postprocessing = False):
    
    if trained_model_checkpoint is not None:
        trained_model = keras.models.load_model(trained_model_checkpoint, custom_objects=custom)

    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/predictions/postprocessed')
        os.makedirs(output_folder, exist_ok=True)
        out_fname = output_folder

    fig, axs = plt.subplots(len(path_list), 4, figsize=(8, len(path_list) * 2))

    # fig, axs = plt.subplots(1, 4, figsize=(10, 6))

    for k, img_path in enumerate(path_list):
        print(img_path)
        img = np.load(img_path)
        pred = get_model_prediction(trained_model, np.expand_dims(img, axis=0))[0, :, :]
        pred_post = predict_mask(trained_model, img, threshold=minimum_fascicle_area, coeff_list=watershed_coeff)

        axs[k, 0].imshow(img)

        axs[k, 1].imshow(pred, cmap='gray')

        axs[k, 2].imshow(pred_post, cmap='gray')

        axs[k, 3].imshow(img)
        axs[k, 3].imshow(pred_post, cmap='gray', alpha=0.5)

        for i in range(4):
            axs[k, i].xaxis.set_major_locator(ticker.NullLocator())
            axs[k, i].yaxis.set_major_locator(ticker.NullLocator())

    axs[0, 0].set_title('Original Image')
    axs[0, 1].set_title('Prediction before\n postprocesing')
    axs[0, 2].set_title('Prediction after\n postprocessing')
    axs[0, 3].set_title('Prediction overlayed\n onto image')


    if save:
        plt.savefig(out_fname + '/predictions_with_postprocessing.png')
    if show:
        plt.show()

# To plot the model losses and metrics
def plot_model_losses_and_metrics(loss_filepath, model_name, save=False, show=True):

    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/training_metrics')
        os.makedirs(output_folder, exist_ok=True)
        out_fname = output_folder

    with open(loss_filepath, 'rb') as loss_file:
        model_details = pickle.load(loss_file)
    fig, axs = plt.subplots(2, figsize=(5,5))
    best = np.argmin(model_details['val_loss'])
    print('\nBest at epoch: ', best)
    # IoU
    axs[0].plot(model_details['iou_score'], label = 'Training set')
    axs[0].plot(model_details['val_iou_score'], label = 'Validation set')
    axs[0].set_ylabel("IoU")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylim([0,1])
    print('\nIoU:\ntraining = ', model_details['iou_score'][best], '\nvalidation = ', model_details['val_iou_score'][best])
    # SparseCategoricalAccuracy
    axs[1].plot(model_details['sparse_categorical_accuracy'], label = 'Training set')
    axs[1].plot(model_details['val_sparse_categorical_accuracy'], label = 'Validation set')
    axs[1].set_ylabel("CA")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylim(top = 1)
    axs[1].legend(loc = 'lower right')
    print('\nCategorical Accuracy:\ntraining = ', model_details['sparse_categorical_accuracy'][best], '\nvalidation = ', model_details['val_sparse_categorical_accuracy'][best])
    # SparseCategoricalCrossentropy
    print('\nBCE:\ntraining = ', model_details['sparse_categorical_crossentropy'][best], '\nvalidation = ',
          model_details['val_sparse_categorical_crossentropy'][best])
    # Focal Loss
    print('\nFocal Loss:\ntraining = ', model_details['sigmoid_focal_crossentropy'][best], '\nvalidation = ',
          model_details['val_sigmoid_focal_crossentropy'][best])
    plt.tight_layout()
    
    if save:
        plt.savefig(out_fname + '/training_metrics_' + model_name + '.png')
    if show:
        plt.show()

if __name__ == '__main__':
    initialise_run()
    train_folder = os.path.join(os.getcwd(), 'data/vagus_dataset_11/train')
    validation_folder = os.path.join(os.getcwd(), 'data/vagus_dataset_11/validation')
    unlabelled_folder = os.path.join(os.getcwd(), 'data/vagus_dataset_11/unlabelled')
    transfer_folder = os.path.join(os.getcwd(), 'data/transfer_learning/train')

    model_save_file = os.path.join(os.getcwd(), model_path)

    ##################### Show Model Metrics ##########################################
    plot_model_losses_and_metrics('model_losses/BCE_Adam_default.pkl', 'BCE', save=True, show=True)
    plot_model_losses_and_metrics('model_losses/FL_Adam_default.pkl', 'FL', save=True, show=True)
    plot_model_losses_and_metrics('model_losses/FL_and_BCE_Adam_default.pkl', 'FL+BCE', save=True, show=True)

    ##################### Show augmented images ##########################################
    sample_img_path = get_samples(train_folder, test=True)
    sample_img_path = sample_img_path[0]
    plot_augmented_images(sample_img_path, num_aug=0, num_aug_wcolor=6, save=True, show=True)

    ##################### Show mask vs prediction #######################################
    path_list = get_samples(validation_folder, num_samples=3)
    plot_masks_vs_predictions(path_list=path_list, trained_model_checkpoint=model_save_file, wstats=True, save=True, show=True)
    plot_masks_vs_predictions(path_list=path_list, trained_model_checkpoint=model_save_file, save=True, show=True)

    ##################### Show post processed prediction #############################################
    unlabelled_sample_list = get_samples(unlabelled_folder, test=True, num_samples=3)
    plot_postprocessed(path_list=unlabelled_sample_list, trained_model_checkpoint=model_save_file, save=True, show=True)

    #################### Show distributions #############################################
    sample_train = get_samples(train_folder, num_samples=-1)
    plot_fascicles_distribution(sample_train, trained_model_checkpoint=model_save_file, save=True, show=True, postprocessing = False)
    plot_fascicles_distribution(sample_train, trained_model_checkpoint=model_save_file, save=True, show=True, postprocessing=True)



