#!/usr/bin/env python3
import pickle
import datetime
import numpy as np
import logging
import os
import json
import heapq

log_path = r"C:\Arbeit\Phoenix\LT-seq2seq\src\logs"
os.makedirs(log_path, exist_ok=True)
log_file = os.path.join(log_path, f"seq2seq_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from tensorflow import keras
except Exception as e:
    logger.warning(f"Failed to import tensorflow.keras: {e}")
    import keras

# Batch size: Size of samples to train the model on.
batch_size = 128  # 64
# Number of epochs to train for.
epochs = 100
# Latent dimensionality of the encoding space (per direction for the BiLSTM).
latent_dim = 512  # prefer 256, 512 over 1024
# Number of samples to train on.
set_num_samples = 50000  # max: 277891

# encoder_path = r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\encoder_model.keras"
# decoder_path = r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\decoder_model.keras"
# Base paths (we'll append direction suffixes)
trained_models_dir = r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models"
os.makedirs(trained_models_dir, exist_ok=True)

# Path to the data txt file on disk.
data_path = r"C:\Arbeit\Phoenix\LT-seq2seq\data\deu.txt"


# Eval funcs.
def train_test_split_indices(n_samples, test_ratio=0.2, seed=42):
    """
    Deterministically split indices into train and test indices.
    Returns (train_idx, test_idx) as numpy arrays.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    test_size = int(n_samples * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return train_idx, test_idx


def levenshtein(a: str, b: str) -> int:
    """
    Compute Levenshtein edit distance between two strings a and b.
    Using dynamic programming.
    """
    la = len(a)
    lb = len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = np.zeros((la + 1, lb + 1), dtype=int)
    for i in range(la + 1):
        dp[i, 0] = i
    for j in range(lb + 1):
        dp[0, j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # deletion
                dp[i, j - 1] + 1,      # insertion
                dp[i - 1, j - 1] + cost  # substitution
            )
    return int(dp[la, lb])


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Compute CER = edit distance / length(reference).
    If reference is empty, return 0.0 to avoid division by zero.
    """
    ref = reference or ""
    hyp = hypothesis or ""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    dist = levenshtein(ref, hyp)
    return dist / len(ref)


def exact_match_accuracy(results):
    """
    results: iterable of dicts with keys 'prediction' and 'reference'
    """
    if len(results) == 0:
        return 0.0
    correct = sum(1 for r in results if r['prediction'] == r['reference'])
    return correct / len(results)


def average_cer(results):
    """
    results: iterable of dicts with keys 'prediction' and 'reference'
    """
    if len(results) == 0:
        return 0.0
    cers = [character_error_rate(r['reference'], r['prediction']) for r in results]
    return float(np.mean(cers))


def save_predictions(results, filename):
    """
    Save list of dicts to JSON Lines (one JSON per line).
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Predictions saved to {filename}")


def save_metrics(metrics: dict, filename: str):
    """
    Save metrics dict as pretty JSON.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics saved to {filename}")


class DataPreprocessor:
    """
    Prepares data for training / inference:
    - extractChar: read and produce char sets and text lists
    - encodingChar: one-hot encodings for inputs/outputs
    - prepareData: convenience wrapper to run the two above
    """

    def __init__(self, data_path: str = data_path, num_samples: int = set_num_samples):
        logger.info(f"Initializing DataPreprocessor with data_path={data_path}, num_samples={num_samples}")
        self.data_path = data_path
        self.num_samples = num_samples

    def prepareData(self, exchangeLanguage: bool = False):
        """
        Prepares data for training a sequence-to-sequence model by extracting characters,
        encoding them into one-hot vectors, and organizing necessary data structures.
        """
        logger.info("Starting prepareData method")
        input_characters, target_characters, input_texts, target_texts = self._extractChar(
            self.data_path, exchangeLanguage=exchangeLanguage, num_samples=self.num_samples
        )
        logger.info(f"Extracted {len(input_texts)} samples from data")
        encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length = self._encodingChar(
            input_characters, target_characters, input_texts, target_texts
        )

        logger.info(f"Encoding completed: {num_encoder_tokens} encoder tokens, {num_decoder_tokens} decoder tokens")
        return (
            encoder_input_data,
            decoder_input_data,
            decoder_target_data,
            input_token_index,
            target_token_index,
            input_texts,
            target_texts,
            num_encoder_tokens,
            num_decoder_tokens,
            max_encoder_seq_length,
        )

    def _extractChar(self, data_path, exchangeLanguage=False, num_samples=None):
        """
        Extracts characters and texts from a language translation dataset.
        """
        logger.info(f"Starting extractChar method with data_path={data_path}, exchangeLanguage={exchangeLanguage}, num_samples={num_samples}")
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()

        if num_samples is None:
            num_samples = 100

        with open(data_path, encoding='utf-8') as f:
            lines = f.read().split('\n')
            logger.info(f"Total lines in file: {len(lines) - 1}")

            if exchangeLanguage == False:
                for line in lines[: min(num_samples, len(lines) - 1)]:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        input_text, target_text = parts[:2]
                        target_text = '\t' + target_text + '\n'
                        input_texts.append(input_text)
                        target_texts.append(target_text)
                        input_characters.update(input_text)
                        target_characters.update(target_text)

            else:
                for line in lines[: min(num_samples, len(lines) - 1)]:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        target_text, input_text = parts[:2]
                        target_text = '\t' + target_text + '\n'
                        input_texts.append(input_text)
                        target_texts.append(target_text)
                        input_characters.update(input_text)
                        target_characters.update(target_text)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        logger.info('input_characters: %s', input_characters)
        logger.info('target_characters: %s', target_characters)
        logger.info('Number of input_texts: %d', len(input_texts))
        logger.info('Number of target_texts: %d', len(target_texts))
        logger.debug('First 100 input_texts: %s', input_texts[:100])
        logger.debug('First 100 target_texts: %s', target_texts[:100])

        return input_characters, target_characters, input_texts, target_texts

    def _encodingChar(self, input_characters, target_characters, input_texts, target_texts):
        """
        Encodes input and target texts into one-hot vectors for training a sequence-to-sequence model.
        """
        logger.info("Starting encodingChar method")
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        logger.info('Number of num_encoder_tokens: %d', num_encoder_tokens)
        logger.info('Number of samples: %d', len(input_texts))

        logger.info('Number of unique input tokens: %d', num_encoder_tokens)
        logger.info('Number of unique output tokens: %d', num_decoder_tokens)
        logger.info('Max sequence length for inputs: %d', max_encoder_seq_length)
        logger.info('Max sequence length for outputs: %d', max_decoder_seq_length)

        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
        logger.debug(f"Created input_token_index with {len(input_token_index)} entries")
        logger.debug(f"Created target_token_index with {len(target_token_index)} entries")

        encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
        decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
        logger.info(f"Initialized data arrays: encoder_input_data shape={encoder_input_data.shape}, decoder_input_data shape={decoder_input_data.shape}, decoder_target_data shape={decoder_target_data.shape}")

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.0
            # Pad encoder input sequences with the padding token
            encoder_input_data[i, len(input_text):, input_token_index[" "]] = 1.0

            for t, char in enumerate(target_text):
                decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
            # Pad decoder input and target sequences with the padding token
            decoder_input_data[i, len(target_text):, target_token_index[" "]] = 1.0
            decoder_target_data[i, len(target_text):, target_token_index[" "]] = 1.0

            if i % 1000 == 0 and i > 0:
                logger.debug(f"Processed {i} samples")

        logger.debug('encoder_input_data sample shape: %s', encoder_input_data.shape)
        logger.debug('decoder_target_data sample shape: %s', decoder_target_data.shape)
        logger.info("Encoding completed successfully")
        return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length


class Seq2SeqModel:
    """
    (uses Bidirectional encoder + decoder) 
    Model handling: construct training model, train it, build inference models, encode/decode sequences,
    and save/load token encodings.
    """

    def __init__(self, latent_dim_local=latent_dim):
        logger.info(f"Initializing Seq2SeqModel with latent_dim={latent_dim_local}")
        self.latent_dim = latent_dim_local

    def modelTranslation(self, num_encoder_tokens, num_decoder_tokens):
        """
        Defines a sequence-to-sequence translation model architecture using a Bidirectional LSTM encoder and LSTM decoder.
        """
        logger.info(f"Building modelTranslation with num_encoder_tokens={num_encoder_tokens}, num_decoder_tokens={num_decoder_tokens}")
        # Encoder
        encoder_inputs = keras.Input(shape=(None, num_encoder_tokens), name="encoder_inputs")
        # Use a named Bidirectional wrapper so we can find it in saved model
        encoder_lstm = keras.layers.LSTM(self.latent_dim, return_state=True, dropout=0.3, recurrent_dropout=0.3, name="encoder_lstm")
        encoder_bi = keras.layers.Bidirectional(encoder_lstm, name="encoder_bilstm", merge_mode="concat")
        # When return_state=True on the wrapped LSTM, the Bidirectional wrapper returns:
        # [output, forward_h, forward_c, backward_h, backward_c]
        encoder_outputs_and_states = encoder_bi(encoder_inputs)
        # unpack, encoder_outputs_and_states is a list/tuple: output, f_h, f_c, b_h, b_c
        encoder_output = encoder_outputs_and_states[0]
        forward_h = encoder_outputs_and_states[1]
        forward_c = encoder_outputs_and_states[2]
        backward_h = encoder_outputs_and_states[3]
        backward_c = encoder_outputs_and_states[4]

        # Concatenate forward and backward states to form initial decoder states
        state_h = keras.layers.Concatenate(name="encoder_h_concat")([forward_h, backward_h])
        state_c = keras.layers.Concatenate(name="encoder_c_concat")([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        # Decoder
        # Because encoder states are concatenated, the decoder needs 2 * latent_dim units
        decoder_units = self.latent_dim * 2
        decoder_inputs = keras.Input(shape=(None, num_decoder_tokens), name="decoder_inputs")
        decoder_lstm = keras.layers.LSTM(decoder_units, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.3, name="decoder_lstm")
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax', name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_bi_model")
        logger.info("Model built successfully")
        model.summary(print_fn=logger.info)

        return model, decoder_outputs, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense

    def trainSeq2Seq(self, model, encoder_input_data, decoder_input_data, decoder_target_data, d=""):
        """
        Trains, logs and saves the sequence-to-sequence model for translation.

        Changes: uses Adam optimizer with gradient clipping, EarlyStopping, ReduceLROnPlateau,
        and saves training history.
        """
        logger.info("Starting trainSeq2Seq method")
        log_dir = r"C:\Arbeit\Phoenix\LT-seq2seq\src\logs\fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.info(f"TensorBoard log directory: {log_dir}")
        tboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        early_stop_callback = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )

        # Adam with gradient clipping
        optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("Model compiled with Adam optimizer (clipnorm=1.0) and categorical_crossentropy loss")

        logger.info(f"Starting training with batch_size={batch_size}, epochs={epochs}")
        history = model.fit([encoder_input_data, decoder_input_data],
                            decoder_target_data,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2,
                            callbacks=[tboard_callback, early_stop_callback, reduce_lr])

        # Save history for diagnostics
        history_path = os.path.join("experiments", "last_training_history.json")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(history.history, fh, indent=2)
        logger.info(f"Training history saved to {history_path}")

        # best epoch info reported
        if "val_loss" in history.history and len(history.history["val_loss"]) > 0:
            best_epoch = int(np.argmin(history.history["val_loss"])) + 1
            logger.info(f"selected epoch to stop training (perhaps EarlyStopping): {best_epoch}")
            logger.info(f"Best val_loss: {min(history.history['val_loss']):.4f}")

        # Save full trained model
        save_path = os.path.join(trained_models_dir, f"sq2sq_model_{d}.keras")
        model.save(save_path, overwrite=True)
        logger.info(f"Model saved to {save_path}")
        return history, save_path 

    def generateInferenceModel(self, input_token_index, target_token_index, direction_suffix="deu2eng", full_models_path=None):
        """
        Generates inference models for sequence-to-sequence translation using a pretrained model.
        , adjusted to work with the Bidirectional encoder (it looks up named layers).
        """
        logger.info("Starting generateInferenceModel method")
        # restore the model and construct (sampling models) the encoder and decoder.
        # model_path = r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\sq2sq_model.keras"
        # model_path = os.path.join(trained_models_dir, f"sq2sq_model.keras")
        logger.info(f"Loading model from {full_models_path}")
        model = keras.models.load_model(full_models_path)
        logger.info("Model loaded for inference construction")

        # Finding encoder bidirectional layer and extracting its outputs (output, f_h, f_c, b_h, b_c)
        try:
            encoder_bi_layer = model.get_layer("encoder_bilstm")
        except Exception:
            logger.exception("Could not find layer named 'encoder_bilstm' in the model. Layer names in the saved model:")
            logger.info([l.name for l in model.layers])
            raise

        # encoder inputs tensor
        encoder_inputs = model.input[0]

        # encoder_bi_layer.output should be a list/tuple: (output, f_h, f_c, b_h, b_c)
        enc_outputs = encoder_bi_layer.output
        try:
            encoder_output = enc_outputs[0]
            forward_h = enc_outputs[1]
            forward_c = enc_outputs[2]
            backward_h = enc_outputs[3]
            backward_c = enc_outputs[4]
        except Exception:
            logger.exception("Unexpected encoder_bilstm.output structure; check the training model setup.")
            raise

        # Concatenate forward/backward states to create the encoder states for inference
        state_h = keras.layers.Concatenate(name="inf_encoder_h")([forward_h, backward_h])
        state_c = keras.layers.Concatenate(name="inf_encoder_c")([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        encoder_model = keras.Model(encoder_inputs, encoder_states)
        logger.info("Encoder inference model constructed")

        # decoder inference model
        # decoder expects decoder input plus two state inputs (h and c), each of size latent_dim*2
        decoder_inputs = model.input[1]

        # shapes for state inputs must match the decoder LSTM's state size
        decoder_state_input_h = keras.Input(shape=(self.latent_dim * 2,), name="decoder_state_input_h")
        decoder_state_input_c = keras.Input(shape=(self.latent_dim * 2,), name="decoder_state_input_c")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        # locate decoder LSTM and dense by name
        try:
            decoder_lstm_layer = model.get_layer("decoder_lstm")
            decoder_dense_layer = model.get_layer("decoder_dense")
        except Exception:
            logger.exception("Could not find 'decoder_lstm' or 'decoder_dense' layers. Layer names:")
            logger.info([l.name for l in model.layers])
            raise

        # use the decoder LSTM layer to get outputs given inputs and states
        decoder_outputs_and_states = decoder_lstm_layer(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_outputs = decoder_outputs_and_states[0]
        state_h_dec = decoder_outputs_and_states[1]
        state_c_dec = decoder_outputs_and_states[2]
        decoder_states = [state_h_dec, state_c_dec]

        decoder_outputs = decoder_dense_layer(decoder_outputs)
        decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        logger.info("Decoder inference model constructed")

        # to store resp. indices for given char. sequences:
        reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

        # # inference models saved
        # encoder_model.save(encoder_path, overwrite=True)
        # decoder_model.save(decoder_path, overwrite=True)
        # logger.info(f"Inference models saved: encoder to {encoder_path}, decoder to {decoder_path}")
        # save inference models with direction suffix
        encoder_out = os.path.join(trained_models_dir, f"encoder_model_{direction_suffix}.keras")
        decoder_out = os.path.join(trained_models_dir, f"decoder_model_{direction_suffix}.keras")
        encoder_model.save(encoder_out, overwrite=True)
        decoder_model.save(decoder_out, overwrite=True)
        logger.info(f"Inference models saved: encoder to {encoder_out}, decoder to {decoder_out}")

        return encoder_model, decoder_model, reverse_target_char_index

    def saveChar2encoding(self, filename, input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index):
        """
        Saves the encodings required for inference.
        """
        # To save as a dictionary:
        # data = {
        #     'input_token_index': input_token_index,
        #     'max_encoder_seq_length': max_encoder_seq_length,
        #     'num_encoder_tokens': num_encoder_tokens,
        #     'reverse_target_char_index': reverse_target_char_index,
        #     'num_decoder_tokens': num_decoder_tokens,
        #     'target_token_index': target_token_index
        # }
        # with open(filename, "wb") as f:
        #     pickle.dump(data, f)

        logger.info(f"Saving character encodings to {filename}")
        with open(filename, "wb") as f:
            pickle.dump(input_token_index, f)
            pickle.dump(max_encoder_seq_length, f)
            pickle.dump(num_encoder_tokens, f)
            pickle.dump(reverse_target_char_index, f)
            pickle.dump(num_decoder_tokens, f)
            pickle.dump(target_token_index, f)
        logger.info(f"Character encodings saved successfully to {filename}")
 
    def decode_sequence(self, input_seq, encoder_model, decoder_model, num_decoder_tokens, target_token_index, reverse_target_char_index, max_decoder_seq_length_given):
        """
        (Greedy) Decode a sequence from the input sequence using the trained encoder and decoder models.
        """
        logger.info("Starting decode_sequence method")
        states_value = encoder_model.predict(input_seq, verbose=0)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, target_token_index['\t']] = 1.0

        stop_condition = False
        decoded_sentence = ""
        step = 0
        while not stop_condition:
            outputs_and_states = decoder_model.predict([target_seq] + states_value, verbose=0)
            output_tokens = outputs_and_states[0]
            h = outputs_and_states[1]
            c = outputs_and_states[2]

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length_given):
                stop_condition = True

            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0
            states_value = [h, c]
            step += 1
            if step % 10 == 0:
                logger.debug(f"Decoding step {step}, current sentence: {decoded_sentence}")

        logger.info(f"Decoded sequence: {decoded_sentence}")
        return decoded_sentence

    def decode_sequence_beam(self, input_seq, encoder_model, decoder_model,
                             num_decoder_tokens, target_token_index,
                             reverse_target_char_index, max_decoder_seq_length_given=51,
                             beam_width=3): 
        """
        Decode a sequence using beam search instead of greedy decoding.
        , keeps the top `beam_width` most probable sequences at each step and expands them.
        """
        states_value = encoder_model.predict(input_seq, verbose=0)
        beam = [(0.0, '\t', states_value)]
        completed = []
        for step in range(max_decoder_seq_length_given):
            new_beam = []
            for log_prob, seq, states in beam:
                target_seq = np.zeros((1, 1, num_decoder_tokens))
                last_char = seq[-1]
                if last_char not in target_token_index:
                    continue
                target_seq[0, 0, target_token_index[last_char]] = 1.0
                outputs_and_states = decoder_model.predict([target_seq] + states, verbose=0)
                output_tokens = outputs_and_states[0]
                new_states = outputs_and_states[1:]

                top_indices = np.argsort(output_tokens[0, -1, :])[-beam_width:]
                for idx in top_indices:
                    char = reverse_target_char_index[int(idx)]
                    prob = float(output_tokens[0, -1, idx])
                    new_log_prob = log_prob + np.log(prob + 1e-12)
                    new_seq = seq + char
                    if char == '\n':
                        completed.append((new_log_prob, new_seq))
                    else:
                        new_beam.append((new_log_prob, new_seq, new_states))
            beam = heapq.nlargest(beam_width, new_beam, key=lambda x: x[0])
            if not beam:
                break

        if completed:
            best_seq = max(completed, key=lambda x: x[0])[1]
        elif beam:
            best_seq = max(beam, key=lambda x: x[0])[1]
        else:
            best_seq = ''
        return best_seq.strip('\t').split('\n')[0]

    def encodingSentenceToPredict(self, sentence, input_token_index, max_encoder_seq_length, num_encoder_tokens):
        """
        Encode a given sentence into its corresponding one-hot encoding based on the provided token index.
        """
        logger.info(f"Encoding sentence for prediction: '{sentence}'")
        encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        for t, char in enumerate(sentence):
            if t < max_encoder_seq_length:
                if char in input_token_index:
                    encoder_input_data[0, t, input_token_index[char]] = 1.0
                else:
                    logger.warning(f"Character '{char}' not found in input_token_index, skipping")
        logger.debug(f"Encoded sentence shape: {encoder_input_data.shape}")
        return encoder_input_data

    def getChar2encoding(self, filename):
        """
        Retrieve trained weights and indices needed for inference from the pickle file.
        """
        logger.info(f"Loading character encodings from {filename}")
        with open(filename, "rb") as f:
            input_token_index = pickle.load(f)
            max_encoder_seq_length = pickle.load(f)
            num_encoder_tokens = pickle.load(f)
            reverse_target_char_index = pickle.load(f)
            num_decoder_tokens = pickle.load(f)
            target_token_index = pickle.load(f)

        logger.info(f"Character encodings loaded successfully: {len(input_token_index)} input tokens, {len(target_token_index)} target tokens")
        return input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index


def run_pipeline(direction: str = "deu2eng", data_path_local: str = data_path, char2enc_dir: str = trained_models_dir): #r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\char2encoding.pkl"):
    """Runs the full pipeline to prepare data, build, train, and save the sequence-to-sequence model and encodings. Possible direction of translations: 'deu2eng' (German->English) or 'eng2deu' (English->German)"""
    assert direction in ("deu2eng", "eng2deu"), "direction can only be 'deu2eng' or 'eng2deu'"
    logger.info("-" * 60)
    logger.info(f"STARTING SEQ2SEQ PIPELINE (direction={direction})")
    logger.info("-" * 60)

    # Prepare data
    logger.info("Step 1: Preparing data")
    exchangeLanguage = True if direction == "eng2deu" else False
    prep = DataPreprocessor(data_path=data_path_local, num_samples=set_num_samples)
    (
        encoder_input_data,
        decoder_input_data,
        decoder_target_data,
        input_token_index,
        target_token_index,
        input_texts,
        target_texts,
        num_encoder_tokens,
        num_decoder_tokens,
        max_encoder_seq_length,
    ) = prep.prepareData(exchangeLanguage=exchangeLanguage)

    # compute max_decoder_seq_length from target_texts (was available during encoding but not returned)
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    logger.info(f"Computed max_decoder_seq_length: {max_decoder_seq_length}")

    # create deterministic train/test split (indices)
    n_samples = encoder_input_data.shape[0]
    logger.info(f"Total number of samples available for split: {n_samples}")
    train_idx, test_idx = train_test_split_indices(n_samples, test_ratio=0.2, seed=42)
    logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    # Split the prepared arrays into training subsets and keep test indexes for evaluation
    encoder_input_data_train = encoder_input_data[train_idx]
    decoder_input_data_train = decoder_input_data[train_idx]
    decoder_target_data_train = decoder_target_data[train_idx]
    logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    logger.info(f"Training arrays shapes after split - encoder: {encoder_input_data_train.shape}, decoder_input: {decoder_input_data_train.shape}, decoder_target: {decoder_target_data_train.shape}")

    # Build the sequence-to-sequence model
    logger.info("Step 2: Building model")
    model_tool = Seq2SeqModel(latent_dim_local=latent_dim)
    model, decoder_outputs, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense = model_tool.modelTranslation(
        num_encoder_tokens, num_decoder_tokens
    )

    # Train the sequence-to-sequence model and save as 'sq2sq_model.keras'
    logger.info("Step 3: Training model (and saving full model with direction suffix)")
    history, sq2sq_save = model_tool.trainSeq2Seq(model, encoder_input_data_train, decoder_input_data_train, decoder_target_data_train, d=direction)

    # After training, rename/save full model w/ trained direction
    sq2sq_save2 = os.path.join(trained_models_dir, f"sq2sq_model_{direction}_backup.keras")
    model.save(sq2sq_save2, overwrite=True)
    logger.info(f"Saved full seq2seq model backup (with direction) to {sq2sq_save2}")

    # encoder-decoder inference model built and saved
    logger.info("Step 4: Building inference models from the saved directional model")
    encoder_model, decoder_model, reverse_target_char_index = model_tool.generateInferenceModel(input_token_index, target_token_index, direction_suffix=direction, full_models_path=sq2sq_save)

    # Save char2encoding with direction suffix
    logger.info("Step 5: Saving character encodings") 
    char2enc_file = os.path.join(char2enc_dir, f"char2encoding_{direction}.pkl")
    model_tool.saveChar2encoding(char2enc_file, input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index)
    logger.info(f"Saved char2encoding to {char2enc_file}")

    logger.info("Step 6: Evaluating model on held-out test set (using beam search)")
    results = []
    for idx in test_idx:
        src = input_texts[idx]
        tgt = target_texts[idx]
        # reference: strip start and end tokens for metric calculation
        reference = tgt.strip()  # removes leading \t and trailing \n
        # prepare input
        input_seq = model_tool.encodingSentenceToPredict(src, input_token_index, max_encoder_seq_length, num_encoder_tokens)
        prediction_raw = model_tool.decode_sequence_beam(input_seq, encoder_model, decoder_model, num_decoder_tokens, target_token_index, reverse_target_char_index, max_decoder_seq_length, beam_width=3)
        prediction = prediction_raw.strip()
        results.append({
            "index": int(idx),
            "source": src,
            "reference": reference,
            "prediction": prediction
        })

    # computing metrics
    exact_acc = exact_match_accuracy(results)
    avg_cer = average_cer(results)
    pred_lengths = [len(r["prediction"]) for r in results]
    ref_lengths = [len(r["reference"]) for r in results]
    truncations = sum(1 for l in pred_lengths if l >= max_decoder_seq_length)
    metrics = {
        "direction": direction,
        "exact_match_accuracy": exact_acc,
        "average_CER": avg_cer,
        "num_test_samples": len(results),
        "avg_pred_length": float(np.mean(pred_lengths)) if pred_lengths else 0.0,
        "avg_ref_length": float(np.mean(ref_lengths)) if ref_lengths else 0.0,
        "truncation_count": int(truncations),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    logger.info("Evaluation metrics:")
    logger.info(json.dumps(metrics, indent=2))

    # Save experiment results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    exp_name = f"char_lstm_bi_{direction}_{timestamp}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    preds_file = os.path.join(exp_dir, "predictions.jsonl")
    metrics_file = os.path.join(exp_dir, "metrics.json")
    save_predictions(results, preds_file)
    save_metrics(metrics, metrics_file)

    logger.info("-" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("-" * 60)
    logger.info("Trained models, encodings, and evaluation saved.")
    logger.info(f"Experiment directory: {os.path.abspath(exp_dir)}")
    return {
        "history": history.history if 'history' in locals() else None,
        "metrics": metrics,
        "predictions_file": preds_file,
        "metrics_file": metrics_file,
        "model_files": {
            "sq2sq": sq2sq_save,
            "encoder": os.path.join(trained_models_dir, f"encoder_model_{direction}.keras"),
            "decoder": os.path.join(trained_models_dir, f"decoder_model_{direction}.keras"),
            "char2encoding": char2enc_file
        }
    }

if __name__ == "__main__":
    logger.info("Training initiated.")
    run_pipeline(direction="deu2eng")  # or "eng2deu" 
    logger.info("Training finished.")