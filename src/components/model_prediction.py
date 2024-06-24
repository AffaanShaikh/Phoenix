import keras
import argparse 
from model_utils import getChar2encoding, encodingSentenceToPredict, decode_sequence

filename="char2encoding.pkl" 

def parse_arguments():
    """
    Parses command line argument
    """
    parser = argparse.ArgumentParser(description="English to German!")
    parser.add_argument("-s","--input_sentence", type=str, required=True, help="Input English sentence to translate.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    input_token_index, max_encoder_seq_length, num_encoder_tokens, \
        reverse_target_char_index, num_decoder_tokens, target_token_index = getChar2encoding(filename)

    # Encode the input sentence
    encoder_input_data = encodingSentenceToPredict(args.input_sentence, input_token_index, max_encoder_seq_length, num_encoder_tokens)

    # Load encoder and decoder models
    encoder_model = keras.models.load_model('trained_models/encoder_model.keras')
    decoder_model = keras.models.load_model('trained_models/decoder_model.keras')

    # Predict and decode the sequence
    input_seq = encoder_input_data
    decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, num_decoder_tokens, target_token_index, reverse_target_char_index, max_decoder_seq_length_given=51)
    # Adjust max_decoder_seq_length_given according to 'max_decoder_seq_length'

    print('\n------------------------------------------\n')
    print('Input sentence:', args.input_sentence)
    print('Decoded sentence:', decoded_sentence)

if __name__ == "__main__":
    main()