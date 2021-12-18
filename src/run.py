from config import img_size, num_classes, initialise_run, cur_model_id
from data_utils import input_target_path_pairs
from model import get_model
from fine_tune import fine_tune
from train import train

if __name__ == '__main__':
    initialise_run()
    model, decoder_layers, encoder_layers = get_model(img_size, num_classes)
    print("Decoder Layers: " + ' '.join(str(e) for e in decoder_layers))
    print("Encoder Layers: " + ' '.join(str(e) for e in encoder_layers))
    m = train(
        model=model,
        model_id=cur_model_id,
        train_img_target_pairs=input_target_path_pairs('data/transfer_learning/train'),
        val_img_target_pairs=input_target_path_pairs('data/transfer_learning/validation')
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
