import os

from config import img_size, num_classes, initialise_run, cur_model_id
from data_utils import input_target_path_pairs
from model import get_model
from train import train


def main(data_dir='data/original_dataset', verbose=True):
    """
        :param data_dir specifies dataset directory
        :param verbose flag for printing messages
    """
    initialise_run()
    model, decoder_layers, encoder_layers = get_model(img_size, num_classes)
    if verbose:
        print("Decoder Layers: " + ' '.join(str(e) for e in decoder_layers))
        print("Encoder Layers: " + ' '.join(str(e) for e in encoder_layers))
    m = train(
        model=model,
        model_id=cur_model_id,
        train_img_target_pairs=input_target_path_pairs(os.path.join(data_dir, 'train')),
        val_img_target_pairs=input_target_path_pairs(os.path.join(data_dir, 'validation'))
    )
    # m = fine_tune(
    #     trained_model_path=trained_model_path,
    #     model_id=ft_model_id,
    #     num_blocks_fine_tune=4,
    #     encoder_layers=encoder_layers,
    #     train_img_target_pairs=input_target_path_pairs('data/transfer_learning_dataset/train'),
    #     val_img_target_pairs=input_target_path_pairs('data/transfer_learning_dataset/validation')
    # )

    if verbose:
        print('Done')


if __name__ == '__main__':
    main()
