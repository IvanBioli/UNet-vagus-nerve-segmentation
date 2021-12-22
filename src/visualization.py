import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorflow import keras
import tensorflow_addons as tfa
from loss import dice_loss, nerve_segmentation_loss, tversky_loss, iou_score, focal_tversky_loss, focal_loss, custom_loss
from eval import predict_mask, get_model_prediction
from stats import get_samples, calculate_regions, compute_bins, get_image_histogram, get_dataset_histogram
from augmentation import get_random_affine_transformation
from post_processing import draw_outliers_regions

from config import initialise_run, model_path, minimum_fascicle_area, watershed_coeff

custom = {'iou_score': iou_score, 'dice_loss': dice_loss, 'nerve_segmentation_loss': nerve_segmentation_loss, 'tversky_loss': tversky_loss, 'focal_tversky_loss': focal_tversky_loss, 'SigmoidFocalCrossEntropy': tfa.losses.SigmoidFocalCrossEntropy(), 'focal_loss': focal_loss, 'custom_loss': custom_loss}


def plot_color_histogram(path_list, dataset_path_list, save=False, show=True):
    """
        Plot the color histogram of images agains that of the whole dataset

        Parameters
        ---------------
        path_list: [str]
            paths to the input images
        dataset_path_list: [str]
            path to training dataset folder
        save: bool, optional
            flag for saving the plot
        show: bool, optional
            flag for showing the plot
    """
    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/distributions')
        os.makedirs(output_folder, exist_ok=True)
        out_fname = output_folder

    set_hist = get_dataset_histogram(dataset_path_list)
    xticks = [i for i in range(256)]
    color = ('r', 'g', 'b')

    fig, axs = plt.subplots(len(path_list) + 1, 4, figsize=(10, (len(path_list)+1) * 2))

    # color histogram of original dataset
    axs[0, 0].text(0.5, 0.5, 'Average on\ntraining set', horizontalalignment='center', verticalalignment='center', transform=axs[0, 0].transAxes)
    axs[0, 0].axis('off')
    for i, col in enumerate(color):
        axs[0, i+1].plot(xticks, set_hist[i], color=col)
        axs[0, i+1].set_xlabel('Colour value')
    axs[0, 1].set_ylabel('% of pixels')

    for k, path in enumerate(path_list):
        k = k+1
        img = np.load(path)
        img_hist = get_image_histogram(img)
        axs[k, 0].imshow(img)
        axs[k, 0].axis('off')

        for i, col in enumerate(color):
            axs[k, i+1].plot(xticks, img_hist[i], color=col)
        axs[k, 1].set_ylabel('% of pixels')

    # histogram of rgb channel
    axs[0, 1].set_title('Histogram of\nR channel')
    axs[0, 2].set_title('Histogram of\nG channel')
    axs[0, 3].set_title('Histogram of\nB channel')
    axs[-1, 1].set_xlabel('Colour value')
    axs[-1, 2].set_xlabel('Colour value')
    axs[-1, 3].set_xlabel('Colour value')

    if save:
        plt.savefig(out_fname + '/color_histograms.png')
    if show:
        plt.show()


def plot_augmented_images(img_path, num_aug=4, num_aug_wcolor=2, save=False, show=True):
    """
        Visualize the different augmentations from a given image

        Parameters
        ---------------
        img_path: str
            path to the input image
        num_aug: int, optional
            number of augmentations to be displayed
        num_aug_wcolor: int, optional
            number of augmentations with color transformation to be displayed
        save: bool, optional
            flag for saving the plot
        show: bool, optional
            flag for showing the plot
    """
    img = np.load(img_path)

    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/augmentations')
        os.makedirs(output_folder, exist_ok=True)
        out_fname = output_folder

    # augmented images
    augmented_imgs = []
    augmented_imgs_wcolor = []

    for _ in range(num_aug):
        transform = get_random_affine_transformation()
        augmented_imgs.append(transform(img, do_colour_transform=False))
    
    for _ in range(num_aug_wcolor):
        transform = get_random_affine_transformation()
        augmented_imgs_wcolor.append(transform(img))

    # reshape np array to fit in the plot
    augmented_imgs= np.reshape(augmented_imgs, (2, num_aug // 2, 512, 512, 3))
    augmented_imgs_wcolor = np.reshape(augmented_imgs_wcolor, (2, num_aug_wcolor // 2, 512, 512, 3))
    
    # plot layouts
    # the last columns always belongs to colored transformations
    num_col = (num_aug + num_aug_wcolor) // 2 + 2
    fig, axs = plt.subplots(2, num_col, figsize=(num_col * 3, 6))

    gs = axs[0][0].get_gridspec()

    for row in range(2): 
        for ax in axs[row][0:2]:
            ax.remove()
    
    original_ax = fig.add_subplot(gs[0:2, 0:2])
    original_ax.imshow(img)

    for x in range(2):
        # normal augmentation
        for y in range(2, num_aug // 2 + 2):
            axs[x][y].imshow(augmented_imgs[x][y - 2])
            axs[x][y].axis('off')

        # colored augmentation
        for y_wcolor in range(-num_aug_wcolor // 2, 0):
            axs[x][y_wcolor].imshow(augmented_imgs_wcolor[x][y_wcolor + num_aug_wcolor // 2])
            axs[x][y_wcolor].axis('off')

    plt.axis('off')

    if save:
        plt.savefig(out_fname + '/augmentations_visualization.png')
    if show:
        plt.show()
        

def plot_masks_vs_predictions(path_list, trained_model_checkpoint=None, wstats=False, save=False, show=True):
    """
        Visualize the original images, the annotated masks, and the predicted masks

        Parameters
        ---------------
        path_list: [str]
            paths to the input images
        trained_model_checkpoint: str
            trained model to load and make prediction
        wstats: bool, optional
            flag to display the metrics stats within the plot
        save: bool, optional
            flag for saving the plot
        show: bool, optional
            flag for showing the plot
    """
    if trained_model_checkpoint is not None:
        trained_model = keras.models.load_model(trained_model_checkpoint, custom_objects=custom)

    if wstats:
        sub_folder = 'wstats'
    else:
        sub_folder = 'default'

    fig, axs = plt.subplots(len(path_list), 4, figsize=(7, len(path_list) * 2))

    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/predictions/', sub_folder)
        os.makedirs(output_folder, exist_ok=True)
        out_fname = output_folder

    for k, path in enumerate(path_list):
        img = np.load(path[0])
        mask = np.load(path[1])

        pred = predict_mask(trained_model, img)

        # original image
        axs[k, 0].imshow(img)

        # annotated mask
        axs[k, 1].imshow(mask, cmap='gray', interpolation='none')

        # predicted mask
        axs[k, 2].imshow(pred, cmap='gray', interpolation='none')

        # predicted overlayed on annotated mask
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

def plot_image_vs_predictions(path_list, trained_model_checkpoint=None, save=False, show=True):
    """
        Visualize the original images, the predicted masks and the predicted masks overlayed onto the original image

        Parameters
        ---------------
        path_list: [str]
            paths to the input images
        trained_model_checkpoint: str
            trained model to load and make prediction
        save: bool, optional
            flag for saving the plot
        show: bool, optional
            flag for showing the plot
    """
    if trained_model_checkpoint is not None:
        trained_model = keras.models.load_model(trained_model_checkpoint, custom_objects=custom)

    sub_folder = 'unlabelled'

    fig, axs = plt.subplots(len(path_list), 3, figsize=(6, len(path_list) * 2))

    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/predictions/', sub_folder)
        os.makedirs(output_folder, exist_ok=True)
        out_fname = output_folder

    for k, path in enumerate(path_list):
        img = np.load(path)

        pred = predict_mask(trained_model, img)

        # original image
        axs[k, 0].imshow(img)

        # prediction
        axs[k, 1].imshow(pred, cmap='gray', interpolation='none')

        # orediction overlayed on image
        axs[k, 2].imshow(img)
        axs[k, 2].imshow(pred, cmap='gray', alpha=0.5, interpolation='none')
        for i in range(3):
            axs[k, i].xaxis.set_major_locator(ticker.NullLocator())
            axs[k, i].yaxis.set_major_locator(ticker.NullLocator())

    axs[0, 0].set_title('Input image')
    axs[0, 1].set_title('Prediction')
    axs[0, 2].set_title('Prediction overlayed\n on input image')

    plt.tight_layout()

    if save:
        plt.savefig(out_fname + '/sample_predictions_' + sub_folder + '.png')
    if show:
        plt.show()

def plot_fascicles_distribution(path_list, test=False, trained_model_checkpoint=None, save=False, show=True, postprocessing=False):
    """
        Plot the the distribution of the stats of the train dataset
        Stats including: fascicles' area, number of fascicle, fascicles' eccentricity

        Parameters
        ---------------
        path_list: [str]
            paths to the input images
        test: bool, optional
            flag to plot the distribution of the annotated masks or not
        trained_model_checkpoint: str
            trained model to load and make prediction
        save: bool, optional
            flag for saving the plot
        show: bool, optional
            flag for showing the plot
        postprocessing: bool, optional
            flag to postprocess the prediction or not
    """

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
    areas_post = []
    num_fascicles_post = []
    eccentricity_post = []

    for p in path_list:
        if not test:
            img_path, mask_path = p
            mask = np.load(mask_path)
        else:
            img_path = p
            mask = None

        img = np.load(img_path)

        pred = predict_mask(trained_model, img)
        regions_pred, regions_mask = calculate_regions(pred, mask)
            
        if not test:
            areas_mask = areas_mask + [m.area for m in regions_mask]
            eccentricity_mask = eccentricity_mask + [m.eccentricity for m in regions_mask]
            num_fascicles_mask.append(len(regions_mask))
        areas_pred = areas_pred + [p.area for p in regions_pred]
        eccentricity_pred = eccentricity_pred + [p.eccentricity for p in regions_pred]
        num_fascicles_pred.append(len(regions_pred))

        if postprocessing:
            pred_post = predict_mask(trained_model, img, threshold=minimum_fascicle_area, coeff_list=watershed_coeff)
            regions_post, _ = calculate_regions(pred_post)
            areas_post = areas_post + [p.area for p in regions_post]
            eccentricity_post = eccentricity_post + [p.eccentricity for p in regions_post]
            num_fascicles_post.append(len(regions_post))

    if not postprocessing:
        fig, axs = plt.subplots(1, 3, figsize=(10, 7))
    if postprocessing:
        fig, axs = plt.subplots(1, 6, figsize=(20, 7))

    nbins_area = 30
    nbins_fascicles = 10
    nbins_eccentricity = 30
    if not test:
        range_plots = [0, 3] if postprocessing else [0]
        for i in range_plots:
            # Computing the bins for the areas' histogram and plotting the histogram
            bins_areas = compute_bins(nbins_area, areas_pred + areas_post, areas_mask)
            axs[0 + i].hist(areas_mask, bins=bins_areas, alpha=0.5, label='Ground truth')
            # Computing the bins for the fascicles' histogram and plotting the histogram
            bins_fascicles = compute_bins(nbins_fascicles, num_fascicles_pred + num_fascicles_post, num_fascicles_mask)
            axs[1 + i].hist(num_fascicles_mask, bins=bins_fascicles, alpha=0.5, label='Ground truth')
            bins_eccentricity = compute_bins(nbins_eccentricity, eccentricity_pred + eccentricity_post, eccentricity_mask)
            axs[2 + i].hist(eccentricity_mask, bins=bins_eccentricity, alpha=0.5, label='Ground truth')
    else:
        bins_areas = compute_bins(nbins_area, areas_pred + areas_post)
        bins_fascicles = compute_bins(nbins_fascicles, num_fascicles_pred + num_fascicles_post)
        bins_eccentricity = compute_bins(nbins_eccentricity, eccentricity_pred + eccentricity_post)

    # Histogram of Fascicles Areas for the predictions
    axs[0].hist(areas_pred, bins=bins_areas, alpha=0.5, label='Prediction')
    axs[0].set_xlabel('Areas (pixels)')
    axs[0].set_ylabel('Number of Occurrencies')
    axs[0].legend(loc='upper right')
    axs[0].set_title('Histogram of \nFascicles Areas')

    # Histogram of Number of fascicles Areas for the predictions
    axs[1].hist(num_fascicles_pred, bins=bins_fascicles, alpha=0.5, label='Prediction')
    axs[1].set_xlabel('Number of Fascicles')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Histogram of \nNumber of Fascicles')

    # Histogram of the Eccentricity
    axs[2].hist(eccentricity_pred, bins=bins_eccentricity, alpha=0.5, label='Prediction')
    axs[2].set_xlabel('Eccentricity')
    axs[2].legend(loc='upper right')
    axs[2].set_title('Histogram of \nEccentricity')

    plt.text(.3, .95, 'Before post-processing', fontsize = 'xx-large', transform=fig.transFigure, horizontalalignment='center')
    if postprocessing:
        # Histogram of Fascicles Areas for the predictions with postprocessing
        axs[3].hist(areas_post, bins=bins_areas, alpha=0.5, label='Prediction')
        axs[3].set_xlabel('Areas (pixels)')
        axs[3].legend(loc='upper right')
        axs[3].set_title('Histogram of \nFascicles Areas')

        # Histogram of Number of fascicles Areas for the predictions with postprocessing
        axs[4].hist(num_fascicles_post, bins=bins_fascicles, alpha=0.5, label='Prediction')
        axs[4].set_xlabel('Number of Fascicles')
        axs[4].legend(loc='upper right')
        axs[4].set_title('Histogram of \nNumber of Fascicles')

        # Histogram of the Eccentricity for the predictions with postprocessing
        axs[5].hist(eccentricity_post, bins=bins_eccentricity, alpha=0.5, label='Prediction')
        axs[5].set_xlabel('Eccentricity')
        axs[5].legend(loc='upper right')
        axs[5].set_title('Histogram of \nEccentricity')

        # To have the same scale on the y axis
        for i in range(3):
            axs[i + 3].set_ylim(axs[i].get_ylim())

        plt.text(.7, .95, 'After post-processing', fontsize = 'xx-large', transform=fig.transFigure, horizontalalignment='center')

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

def plot_postprocessed(path_list, trained_model_checkpoint=None, save=False, show=True):
    """
        Visualized the prediction before and after postprocessing

        Parameters
        ---------------
        path_list: [str]
            paths to the input images
        trained_model_checkpoint: str
            trained model to load and make prediction
        save: bool, optional
            flag for saving the plot
        show: bool, optional
            flag for showing the plot
    """
    
    if trained_model_checkpoint is not None:
        trained_model = keras.models.load_model(trained_model_checkpoint, custom_objects=custom)

    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/predictions/postprocessed')
        os.makedirs(output_folder, exist_ok=True)
        out_fname = output_folder

    fig, axs = plt.subplots(len(path_list), 3, figsize=(6, len(path_list) * 2))

    for k, img_path in enumerate(path_list):
        print(img_path)
        img = np.load(img_path)
        pred = get_model_prediction(trained_model, np.expand_dims(img, axis=0))[0, :, :]
        pred_post = predict_mask(trained_model, img, threshold=minimum_fascicle_area, coeff_list=watershed_coeff)
        pred_post = draw_outliers_regions(pred_post, area_threshold=3101, eccen_threshold=[0, 0.95])

        # original image
        axs[k, 0].imshow(img)

        # prediction
        axs[k, 1].imshow(pred, cmap='gray')

        # prediction afeter processing
        axs[k, 2].imshow(pred_post, cmap='gray')

        for i in range(3):
            axs[k, i].xaxis.set_major_locator(ticker.NullLocator())
            axs[k, i].yaxis.set_major_locator(ticker.NullLocator())

    axs[0, 0].set_title('Original Image')
    axs[0, 1].set_title('Prediction before\n postprocessing')
    axs[0, 2].set_title('Prediction after\n postprocessing')


    if save:
        plt.savefig(out_fname + '/predictions_with_postprocessing.png')
    if show:
        plt.show()

# To plot the model losses and metrics
def plot_model_losses_and_metrics(loss_filepath, model_name, save=False, show=True):
    """
        Plot the losses and metrics during training the model

        Parameters
        ---------------
        loss_filepath: str
            path to the saved losses and metrics recorded from the training
        model_name: str
            the name of the trained model
        save: bool, optional
            flag for saving the plot
        show: bool, optional
            flag for showing the plot
    """

    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/training_metrics')
        os.makedirs(output_folder, exist_ok=True)
        out_fname = output_folder

    with open(loss_filepath, 'rb') as loss_file:
        model_details = pickle.load(loss_file)
    fig, axs = plt.subplots(2, figsize=(5,5))
    best = np.argmin(model_details['val_loss'])
    print('\nBest at epoch: ', best+1)
    # IoU
    axs[0].plot(model_details['iou_score'], label = 'Training set')
    axs[0].plot(model_details['val_iou_score'], label = 'Validation set')
    axs[0].set_ylabel("IoU")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylim([0,1])
    axs[0].legend(loc='lower right')
    print('\nIoU:\ntraining = ', model_details['iou_score'][best], '\nvalidation = ', model_details['val_iou_score'][best])
    # DiceLoss
    axs[1].plot(model_details['dice_loss'], label = 'Training set')
    axs[1].plot(model_details['val_dice_loss'], label = 'Validation set')
    axs[1].set_ylabel("DL")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylim([0,1])
    axs[1].legend(loc = 'upper right')
    print('\nDiceLoss:\ntraining = ', model_details['dice_loss'][best], '\nvalidation = ', model_details['val_dice_loss'][best])
    # SparseCategoricalCrossentropy
    print('\nBCE:\ntraining = ', model_details['sparse_categorical_crossentropy'][best], '\nvalidation = ',
          model_details['val_sparse_categorical_crossentropy'][best])
    # Focal Loss
    print('\nFocal Loss:\ntraining = ', model_details['sigmoid_focal_crossentropy'][best], '\nvalidation = ',
          model_details['val_sigmoid_focal_crossentropy'][best])
    plt.tight_layout()
    #for i, j in enumerate(model_details['val_iou_score']):
    #    print(str(i+1) + ': ' + str(j))
    if save:
        plt.savefig(out_fname + '/training_metrics_' + model_name + '.png')
    if show:
        plt.show()

if __name__ == '__main__':
    initialise_run()

    train_folder = os.path.join(os.getcwd(), 'data/original_dataset/train')
    validation_folder = os.path.join(os.getcwd(), 'data/original_dataset/validation')
    unlabelled_folder = os.path.join(os.getcwd(), 'data/original_dataset/unlabelled')

    model_save_file = os.path.join(os.getcwd(), model_path)
    
    ##################### Show Model Metrics ##########################################
    print('\n------------------------------ Training with BCE loss ------------------------------')
    plot_model_losses_and_metrics('model_losses/BCE_Adam_default.pkl', 'BCE', save=True, show=True)
    print('\n--------------------------------- Training with Fl ---------------------------------')
    plot_model_losses_and_metrics('model_losses/FL_Adam_default.pkl', 'FL', save=True, show=True)
    print('\n------------------------------- Training with BCE+FL -------------------------------')
    plot_model_losses_and_metrics('model_losses/FL_and_BCE_Adam_default.pkl', 'FL+BCE', save=True, show=True)
    print('\n------------------------------- Training with test -------------------------------')
    plot_model_losses_and_metrics('model_losses/test.pkl', 'test', save=True, show=True)
    print('\n------------------------------- Training with test_original -------------------------------')
    plot_model_losses_and_metrics('model_losses/test_original.pkl', 'test_original', save=True, show=True)

    ##################### Show augmented images ###################################################
    sample_img_path = get_samples(train_folder, test=True)
    sample_img_path = sample_img_path[0]
    plot_augmented_images(sample_img_path, num_aug=0, num_aug_wcolor=6, save=True, show=True)

    ##################### Show mask vs prediction ##################################################
    ##### From validation set #####
    path_list = get_samples(validation_folder, num_samples=3)
    plot_masks_vs_predictions(path_list=path_list, trained_model_checkpoint=model_save_file, wstats=True, save=True, show=True)
    ##### From unlabelled set #####
    unlabelled_sample_list = get_samples(unlabelled_folder, test=True, num_samples=15)
    # Picking images to show different behaviours
    unlabelled_sample_list = [unlabelled_sample_list[1], unlabelled_sample_list[11], unlabelled_sample_list[2]]
    plot_image_vs_predictions(path_list=unlabelled_sample_list, trained_model_checkpoint=model_save_file, save=True, show=True)

    ##################### Show color histogram ######################################################
    training_path_list = get_samples(train_folder, test=True, num_samples=-1)
    color_histogram_list = unlabelled_sample_list
    plot_color_histogram(color_histogram_list, training_path_list, save=True, show=True)
    
    ##################### Show histograms before and after postprocessing ############################
    training_path_list = get_samples(train_folder, test=False, num_samples=-1)
    #plot_fascicles_distribution(training_path_list, test=False, trained_model_checkpoint=model_save_file, save=True, show=True, postprocessing=False)
    plot_fascicles_distribution(training_path_list, test=False, trained_model_checkpoint=model_save_file, save=True, show=True, postprocessing=True)

    ##################### Show post processed prediction #############################################
    # Picking images to show different behaviours
    unlabelled_sample_list = [unlabelled_sample_list[0], unlabelled_sample_list[1]]
    plot_postprocessed(path_list=unlabelled_sample_list, trained_model_checkpoint=model_save_file, save=True, show=True)
