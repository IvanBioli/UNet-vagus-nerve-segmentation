from config import img_size, num_classes, initialise_run
from data_utils import input_target_path_pairs
from model import get_model
from src.train import train

if __name__ == '__main__':
    initialise_run()
    model, decoder_layers, encoder_layers = get_model(img_size, num_classes)
    print("Decoder Layers: " + ' '.join(str(e) for e in decoder_layers))
    print("Encoder Layers: " + ' '.join(str(e) for e in encoder_layers))
    cur_model_id = 'Adam_512_SCCE'
    ft_model_id = 'Adam_512_SCCE_fine_tune'
    trained_model_path = 'model_checkpoints\Adam_512_tversky_loss.h5'
    m = train(
        model=model,
        model_id=cur_model_id,
        train_img_target_pairs=input_target_path_pairs('data/vagus_dataset_10/train'),
        val_img_target_pairs=input_target_path_pairs('data/vagus_dataset_10/validation')
    )
    # m = fine_tune(
    #     trained_model_path=trained_model_path,
    #     model_id=ft_model_id,
    #     num_blocks_fine_tune=4,
    #     encoder_layers=encoder_layers,
    #     train_img_target_pairs=input_target_path_pairs('data/transfer_learning_dataset/train'),
    #     val_img_target_pairs=input_target_path_pairs('data/transfer_learning_dataset/validation')
    # )

    # output_predictions(trained_model_id=cur_model_id)
    print('Done')
