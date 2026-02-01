import pickle
import datetime
import numpy as np
import logging
import os
import json

log_path = r"C:\Arbeit\Phoenix\LT-seq2seq\src\logs"
os.makedirs(log_path, exist_ok=True)
log_file = os.path.join(log_path, f"seq2seq_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
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
batch_size = 64  # 128
# Number of epochs to train for.
epochs = 1  # 100
# Latent dimensionality of the encoding space.
latent_dim = 256  # 512 #1024
# Number of samples to train on.
set_num_samples = 10000  # max: 277891

encoder_path = r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\encoder_model.keras"
decoder_path = r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\decoder_model.keras"

# Path to the data txt file on disk.
data_path = r"C:\Arbeit\Phoenix\LT-seq2seq\data\deu.txt"

# -------------------------
# Evaluation / Benchmark helpers
# -------------------------
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
    # convert to lists to allow indexing
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
        # If both empty, CER=0. If ref empty but hyp not, treat as len(hyp)/1 to penalize.
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

    def prepareData(self):
        """
        Prepares data for training a sequence-to-sequence model by extracting characters,
        encoding them into one-hot vectors, and organizing necessary data structures.
        """
        logger.info("Starting prepareData method")
        input_characters, target_characters, input_texts, target_texts = self._extractChar(
            self.data_path, exchangeLanguage=False, num_samples=self.num_samples
        )
        logger.info(f"Extracted {len(input_texts)} samples from data")
        encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length = self._encodingChar(
            input_characters, target_characters, input_texts, target_texts
        )  # extra num_decoder_tokens removed

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

        # encoder tokens: 26 uppercase letters + 26 lowercase letters + 19 punctuations and symbols = 71 encoder tokens
        logger.info('Number of unique input tokens: %d', num_encoder_tokens)

        # decoder tokens: 28 arabic alphabets + 71 all encoder + 43 numbers? things like '\t'
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
            # encoder_input_data[i, t + 1:, input_token_index[" "]] = 1.0 ##

            for t, char in enumerate(target_text):
                decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
            # Pad decoder input and target sequences with the padding token
            decoder_input_data[i, len(target_text):, target_token_index[" "]] = 1.0
            decoder_target_data[i, len(target_text):, target_token_index[" "]] = 1.0
            # decoder_input_data[i, t + 1:, target_token_index[" "]] = 1.0 ##
            # decoder_target_data[i, t:, target_token_index[" "]] = 1.0 ##

            if i % 1000 == 0 and i > 0:
                logger.debug(f"Processed {i} samples")

        logger.debug('encoder_input_data sample shape: %s', encoder_input_data.shape)
        logger.debug('decoder_target_data sample shape: %s', decoder_target_data.shape)
        logger.info("Encoding completed successfully")
        return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length  # extra num_decoder_tokens removed


class Seq2SeqModel:
    """
    Model handling: construct training model, train it, build inference models, encode/decode sequences,
    and save/load token encodings.
    """

    def __init__(self, latent_dim_local=latent_dim):
        logger.info(f"Initializing Seq2SeqModel with latent_dim={latent_dim_local}")
        self.latent_dim = latent_dim_local

    def modelTranslation(self, num_encoder_tokens, num_decoder_tokens):
        """
        Defines a sequence-to-sequence translation model architecture using LSTM layers.
        """
        logger.info(f"Building modelTranslation with num_encoder_tokens={num_encoder_tokens}, num_decoder_tokens={num_decoder_tokens}")
        # Model arcitechture: 1 encoder(lstm) + 1 decode (LSTM) + 1 Dense layer + softmax
        encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))  # Encoder Input layer,'None' represents the variable length of sequences.
        encoder = keras.layers.LSTM(self.latent_dim, return_state=True)  # LSTM layer with latent_dim units
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))  # Decoder Input layer
        decoder_lstm = keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)  # Another LSTM layer with the same number of units as the encoder.
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)

        decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax')  # a Dense layer with num_decoder_tokens units and a softmax activation func.
        decoder_outputs = decoder_dense(decoder_outputs)  # final output of the decoder after applying the dense layer.

        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)  # the Keras Model, which takes encoder and decoder inputs and produces the decoder outputs.
        logger.info("Model built successfully")
        model.summary(print_fn=logger.info)

        return model, decoder_outputs, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense

    def trainSeq2Seq(self, model, encoder_input_data, decoder_input_data, decoder_target_data):
        """
        Trains, logs and saves the sequence-to-sequence model for translation.
        """
        logger.info("Starting trainSeq2Seq method")
        log_dir = r"C:\Arbeit\Phoenix\LT-seq2seq\src\logs\fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.info(f"TensorBoard log directory: {log_dir}")
        tboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("Model compiled with RMSprop optimizer and categorical_crossentropy loss")

        logger.info(f"Starting training with batch_size={batch_size}, epochs={epochs}")
        history = model.fit([encoder_input_data, decoder_input_data],
                            decoder_target_data,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2,
                            callbacks=[tboard_callback])

        save_path = r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\sq2sq_model.keras"
        model.save(save_path, overwrite=True)
        logger.info(f"Model saved to {save_path}")
        logger.info(f"Training completed. Final loss: {history.history['loss'][-1]:.4f}, accuracy: {history.history['accuracy'][-1]:.4f}")

    def generateInferenceModel(self, input_token_index, target_token_index):
        """
        Generates inference models for sequence-to-sequence translation using a pretrained model.
        """
        logger.info("Starting generateInferenceModel method")
        # Restore the model and construct (sampling models) the encoder and decoder.
        model_path = r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\sq2sq_model.keras"
        logger.info(f"Loading model from {model_path}")
        model = keras.models.load_model(model_path)

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("Model loaded and compiled for inference")

        encoder_inputs = model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1]  # input_2
        decoder_state_input_h = keras.Input(shape=(self.latent_dim,))
        decoder_state_input_c = keras.Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]

        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

        # to store resp. indices for given char. sequences:
        reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
        
        encoder_model.save(encoder_path, overwrite=True)
        decoder_model.save(decoder_path, overwrite=True)
        logger.info(f"Inference models saved: encoder to {encoder_path}, decoder to {decoder_path}")

        return encoder_model, decoder_model, reverse_target_char_index

    def saveChar2encoding(self, filename, input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index):
        """
        Saves trained weights and indices needed for inference.
        To save as a dictionary:
        data = {
            'input_token_index': input_token_index,
            'max_encoder_seq_length': max_encoder_seq_length,
            'num_encoder_tokens': num_encoder_tokens,
            'reverse_target_char_index': reverse_target_char_index,
            'num_decoder_tokens': num_decoder_tokens,
            'target_token_index': target_token_index
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        """
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
        Decode a sequence from the input sequence using the trained encoder and decoder models.
        """
        logger.info("Starting decode_sequence method")
        # encode the input.
        states_value = encoder_model.predict(input_seq, verbose=0)
        logger.debug("Encoder prediction completed")

        # Generate empty target sequence of length 1.0
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index['\t']] = 1.0

        stop_condition = False
        decoded_sentence = ""
        step = 0
        # predict the output letter by letter
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)  # , verbose = 0 for silent, 1 for bar, 2 for single line

            # sample the token (it's index) with highest probability, look-up the char for the chosen index and attach it to the decoded seq.
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # check for end of the string
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

    def encodingSentenceToPredict(self, sentence, input_token_index, max_encoder_seq_length, num_encoder_tokens):
        """
        Encode a given sentence into its corresponding one-hot encoding based on the provided token index.
        """
        logger.info(f"Encoding sentence for prediction: '{sentence}'")
        # Initialize the encoder input data with zeros.
        encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')

        for t, char in enumerate(sentence):
            if t < max_encoder_seq_length:  # Ensure we don't exceed the max length.
                if char in input_token_index:  # Ensure the character is in the token index
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


def run_pipeline(data_path_local: str = data_path, char2enc_filename: str = r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\char2encoding.pkl"):
    """Runs the full pipeline to prepare data, build, train, and save the sequence-to-sequence model and encodings."""
    logger.info("=" * 60)
    logger.info("STARTING SEQ2SEQ PIPELINE")
    logger.info("=" * 60)

    # Prepare data
    logger.info("Step 1: Preparing data")
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
    ) = prep.prepareData()

    # compute max_decoder_seq_length from target_texts (was available during encoding but not returned)
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    logger.info(f"Computed max_decoder_seq_length: {max_decoder_seq_length}")

    # create deterministic train/test split (indices)
    n_samples = encoder_input_data.shape[0]
    logger.info(f"Total number of samples available for split: {n_samples}")
    train_idx, test_idx = train_test_split_indices(n_samples, test_ratio=0.001, seed=42)
    logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    # Split the prepared arrays into training subsets and keep test indexes for evaluation
    encoder_input_data_train = encoder_input_data[train_idx]
    decoder_input_data_train = decoder_input_data[train_idx]
    decoder_target_data_train = decoder_target_data[train_idx]
    logger.info(f"Training arrays shapes after split - encoder: {encoder_input_data_train.shape}, decoder_input: {decoder_input_data_train.shape}, decoder_target: {decoder_target_data_train.shape}")

    # Build the sequence-to-sequence model
    logger.info("Step 2: Building model")
    model_tool = Seq2SeqModel(latent_dim_local=latent_dim)
    model, decoder_outputs, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense = model_tool.modelTranslation(
        num_encoder_tokens, num_decoder_tokens
    )

    # Train the sequence-to-sequence model and save as 'sq2sq_model.keras'
    logger.info("Step 3: Training model")
    # NOTE: pass the TRAINING arrays (not the full arrays) to avoid leaking the test samples into training.
    model_tool.trainSeq2Seq(model, encoder_input_data_train, decoder_input_data_train, decoder_target_data_train)

    # Build the encoder-decoder inference model and save it
    logger.info("Step 4: Building inference models")
    encoder_model, decoder_model, reverse_target_char_index = model_tool.generateInferenceModel(input_token_index, target_token_index)

    # Save the character encodings in a pickle file, save object to convert: sequence to encoding and encoding to sequence
    logger.info("Step 5: Saving character encodings")
    model_tool.saveChar2encoding(char2enc_filename, input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index)
 
    logger.info("Step 6: Evaluating model on held-out test set")
    results = []
    for idx in test_idx:
        src = input_texts[idx]
        tgt = target_texts[idx]
        # reference: strip start and end tokens for metric calculation
        reference = tgt.strip()  # removes leading \t and trailing \n
        # prepare input
        input_seq = model_tool.encodingSentenceToPredict(src, input_token_index, max_encoder_seq_length, num_encoder_tokens)
        # decode
        prediction_raw = model_tool.decode_sequence(input_seq, encoder_model, decoder_model, num_decoder_tokens, target_token_index, reverse_target_char_index, max_decoder_seq_length)
        prediction = prediction_raw.strip()
        # Save one example
        results.append({
            "index": int(idx),
            "source": src,
            "reference": reference,
            "prediction": prediction
        })

    # compute metrics
    exact_acc = exact_match_accuracy(results)
    avg_cer = average_cer(results)
    # compute length stats for diagnostic
    pred_lengths = [len(r["prediction"]) for r in results]
    ref_lengths = [len(r["reference"]) for r in results]
    truncations = sum(1 for l in pred_lengths if l >= max_decoder_seq_length)
    metrics = {
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
    exp_name = f"baseline_char_lstm_{timestamp}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    preds_file = os.path.join(exp_dir, "predictions.jsonl")
    metrics_file = os.path.join(exp_dir, "metrics.json")
    # Save raw predictions and metrics
    save_predictions(results, preds_file)
    save_metrics(metrics, metrics_file)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info("Trained models, encodings, and evaluation saved.")
    logger.info(f"Experiment directory: {os.path.abspath(exp_dir)}")


if __name__ == "__main__":
    logger.info("Training initiated.")
    run_pipeline()
    logger.info("Training finished.")
