from config import img_size, num_classes, initialise_run, cur_model_id
from data_utils import input_target_path_pairs
from model import get_model
from train import train

if __name__ == '__main__':
    initialise_run()
    model, decoder_layers, encoder_layers = get_model(img_size, num_classes)
    print("Decoder Layers: " + ' '.join(str(e) for e in decoder_layers))
    print("Encoder Layers: " + ' '.join(str(e) for e in encoder_layers))
    m = train(
        model=model,
        model_id=cur_model_id,
        train_img_target_pairs=input_target_path_pairs('data/original_dataset/train'),
        val_img_target_pairs=input_target_path_pairs('data/original_dataset/validation')
    )
    print('Done')