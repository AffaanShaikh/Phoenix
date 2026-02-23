import streamlit as st
import numpy as np
import keras
import pickle
 
MODEL_DIR = "src/trained_models"

# util func.
@st.cache_resource
def load_models(direction):
    encoder = keras.models.load_model(f"{MODEL_DIR}/encoder_model_{direction}.keras")
    decoder = keras.models.load_model(f"{MODEL_DIR}/decoder_model_{direction}.keras")

    with open(f"{MODEL_DIR}/char2encoding_{direction}.pkl", "rb") as f:
        input_token_index = pickle.load(f)
        max_encoder_seq_length = pickle.load(f)
        num_encoder_tokens = pickle.load(f)
        reverse_target_char_index = pickle.load(f)
        num_decoder_tokens = pickle.load(f)
        target_token_index = pickle.load(f)

    return (
        encoder,
        decoder,
        input_token_index,
        max_encoder_seq_length,
        num_encoder_tokens,
        reverse_target_char_index,
        num_decoder_tokens,
        target_token_index,
    )

# enc.
def encode_input(sentence, input_token_index, max_len, num_tokens):
    x = np.zeros((1, max_len, num_tokens), dtype="float32")
    for t, char in enumerate(sentence):
        if t < max_len and char in input_token_index:
            x[0, t, input_token_index[char]] = 1.0
    return x

# (greedy) decode
def decode_sequence(input_seq, encoder_model, decoder_model,
                    num_decoder_tokens, target_token_index,
                    reverse_target_char_index, max_len=100):

    states_value = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.0

    decoded = ""

    for _ in range(max_len):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )

        idx = np.argmax(output_tokens[0, -1, :])
        char = reverse_target_char_index[idx]

        if char == "\n":
            break

        decoded += char

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, idx] = 1.0

        states_value = [h, c]

    return decoded

# UI
st.set_page_config(page_title="In-house Language Translation", layout="centered")

st.title("Seq2Seq Language Translator (English <-> Deutsch)")
st.markdown("Character-level LSTM-based translator trained on 50k sentence pairs.")

direction = st.radio(
    "Translation Direction",
    ["English to German", "Deutsch zu Englisch"]
)
direction_key = "eng2deu" if direction == "English to German" else "deu2eng"
# load models and token indices
(
    encoder_model,
    decoder_model,
    input_token_index,
    max_len,
    num_encoder_tokens,
    reverse_target_char_index,
    num_decoder_tokens,
    target_token_index,
) = load_models(direction_key)

# Input
user_input = st.text_area("Enter sentence:")

if st.button("Translate"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        input_seq = encode_input(
            user_input,
            input_token_index,
            max_len,
            num_encoder_tokens
        )

        output = decode_sequence(
            input_seq,
            encoder_model,
            decoder_model,
            num_decoder_tokens,
            target_token_index,
            reverse_target_char_index
        )

        st.success(output)