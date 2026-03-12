#!/usr/bin/env python3
""" 
- builds a token-level seq2seq model with a Bidirectional LSTM encoder + attention + LSTM decoder.
    - sq2sq_model_{direction}.keras (full model)
        - encoder_model_{direction}.keras (inference encoder)
        - decoder_model_{direction}.keras (inference decoder)
    - spm model files: spm_{direction}.model, spm_{direction}.vocab
    - char2encoding replaced with token encodings saved as pickle (sp_model path, max lengths, vocab sizes)
- supports full CLI flags now to run training only, eval only, or both :]
""" 
import os, json, pickle, argparse, logging#, heapq 
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import datetime 

log_path = Path(r"C:\Arbeit\Phoenix\LT-seq2seq\src\logs")
log_path.mkdir(parents=True, exist_ok=True)
log_file = log_path / f"seq2seq_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from tensorflow import keras
except Exception as e: 
    logger.warning("Failed to import tensorflow.keras: %s", e)
    import keras 

try:
    import sentencepiece as spm
    HAVE_SPM = True
except Exception:
    HAVE_SPM = False

# hyperparams / defaults
batch_size = 128 # Batch size: Size of samples to train the model on
epochs = 100 # Number of epochs to train for
latent_dim = 512 # prefer 256, 512 over 1024 # LSTM units per direction in encoder (bidirectional), decoder uses latent_dim * 2
set_num_samples = 50000
vocab_size = 8000 # SentencePiece vocab size (subword)
recurrent_dropout = 0.0 # keeping 0.0 to use GPU CuDNN LSTM (fast), set >0 to regularize but GPU slower
use_sentencepiece = True # set False to use word-level tokenization fallback
embedding_dim = 256
dropout = 0.3
# warmup hyparams, warmup_epochs controls how many epochs LR linearly ramps from warmup_initial_lr up to peak_lr before ReduceLROnPlateau takes over
# peak_lr = og Adam lr 1e-3 
warmup_epochs = 5
warmup_initial_lr = 1e-6
peak_lr = 1e-3

trained_models_dir = Path(r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models") # base paths (we'll append direction suffixes)
trained_models_dir.mkdir(parents=True, exist_ok=True)
data_path_default = r"C:\Arbeit\Phoenix\LT-seq2seq\data\deu.txt" # path to the data txt file on disk.
experiments_dir = Path(r"C:\Arbeit\Phoenix\LT-seq2seq\experiments")
experiments_dir.mkdir(parents=True, exist_ok=True)


# WarmUpLRCallback fires on_epoch_begin (before batches), while ReduceLROnPlateau fires on_epoch_end (after batches).
# thereby avoiding conflict, during warmup epochs the callback overwrites whatever RLROP did at the previous epoch-end before any new batches run
# After warmup_epochs, the callback is a no-op and RLROP manages LR from there on, it's beautiful :}
class WarmUpLRCallback(keras.callbacks.Callback):
    """linearly ramps LR from 'warmup_lr' to 'peak_lr' over 'warmup_epochs' epochs"""
    def __init__(self, warmup_epochs: int = 5, peak_lr: float = 1e-3, warmup_lr: float = 1e-6):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.peak_lr = peak_lr
        self.warmup_lr = warmup_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr + (self.peak_lr - self.warmup_lr) * (epoch + 1) / self.warmup_epochs
            # .assign() compatible with both TF Keras and standalone Keras,
            # falls back to keras.backend.set_value for older installations
            try:
                self.model.optimizer.learning_rate.assign(lr)
            except AttributeError:
                keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            logger.info("WarmUpLRCallback epoch %d: lr set to %.2e", epoch, lr)


# Eval func.(s)
class Evaluator:
    @staticmethod
    def train_test_split_indices(n_samples: int, test_ratio: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """splits indices into train and test indices"""
        rng = np.random.default_rng(seed)
        idx = np.arange(n_samples)
        rng.shuffle(idx)
        t = int(n_samples * test_ratio)
        return idx[t:], idx[:t] # train_idx, test_idx

    @staticmethod
    def levenshtein(a: str, b: str) -> int:
        """computes Levenshtein edit distance between two strings a and b using dynamic programming"""
        la, lb = len(a), len(b)
        if la == 0: return lb
        if lb == 0: return la
        dp = np.zeros((la + 1, lb + 1), dtype=int)
        for i in range(la + 1):
            dp[i, 0] = i
        for j in range(lb + 1):
            dp[0, j] = j
        for i in range(1, la + 1):
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
        return int(dp[la, lb])

    @staticmethod
    def cer(ref: str, hyp: str) -> float:
        """CER = edit distance / length(reference)"""
        if len(ref) == 0:
            return 0.0 if len(hyp) == 0 else 1.0
        return Evaluator.levenshtein(ref, hyp) / len(ref)

    @staticmethod
    def exact_match_accuracy(results: List[Dict]) -> float:
        """results: iterable of dicts with keys 'prediction' and 'reference'"""
        if len(results) == 0:
            return 0.0
        correct = sum(1 for r in results if r["prediction"] == r["reference"])
        return correct / len(results)

    @staticmethod
    def average_cer(results: List[Dict]) -> float:
        """avg. cer.. yea"""
        if len(results) == 0:
            return 0.0
        cers = [Evaluator.cer(r["reference"], r["prediction"]) for r in results]
        return float(np.mean(cers))

    @staticmethod
    def save_predictions(results: List[Dict], filename: str) -> None:
        """saves a list of dicts to JSON Lines (one JSON per line)"""
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Predictions saved to %s", filename)

    @staticmethod
    def save_metrics(metrics: Dict, filename: str) -> None:
        """save metrics dict as pretty JSON"""
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("Metrics saved to %s", filename)


class DataPreprocessor:
    """tokenization, encoding, decoding utilities and data prep., supports SentencePiece training & loading, or Keras tokenizer as a fallback"""

    def __init__(self, trained_models_dir: Path, use_sp: bool = True, vocab_size_default: int = 8000):
        self.trained_models_dir = Path(trained_models_dir)
        self.use_sentencepiece = use_sp
        self.vocab_size_default = vocab_size_default

    def trainSentencePiece(self, corpus_texts: List[str], model_prefix: str, vocab_size_local: int) -> Tuple[str, str]:
        """
        trains sentencepiece model from corpus list of strings, writes temporary file for training, returns path to model file
        """
        assert HAVE_SPM, "sentencepiece is not installed. pip install sentencepiece to use subword tokenization."
        tmp_corpus = model_prefix + "_corpus.txt"
        with open(tmp_corpus, "w", encoding="utf-8") as f:
            for s in corpus_texts:
                f.write(s.replace("\n", " ") + "\n") 
        all_text = " ".join(corpus_texts)
        num_unique_chars = len(set(all_text))
        num_special_tokens = 4 # pad, unk, bos, eos
        # smallest sensible vocab must at least contain all unique characters + specials
        min_vocab_size = num_unique_chars + num_special_tokens
        # rough upper bound: each distinct whitespace token could be a piece (plus specials)
        unique_words = len(set(all_text.split()))
        max_vocab_size = unique_words + num_special_tokens
        if max_vocab_size < min_vocab_size:
            max_vocab_size = min_vocab_size
        requested = vocab_size_local
        vocab_size_local = max(min_vocab_size, min(requested, max_vocab_size))
        logger.info("Using vocab_size_local=%d (requested=%d, min=%d, max=%d)",
                    vocab_size_local, requested, min_vocab_size, max_vocab_size)
        # if the effective vocab is less than half of what was requested coz a silent >50% cut would degrade subword quality significantly
        if vocab_size_local < requested * 0.5: # i.e. if < 50% of requested vocab size  
            logger.warning(
                "!!! VOCAB SIZE CUT BY >50%%: requested=%d but effective vocab_size=%d "
                "(min=%d, max=%d). Subword quality may be poor. Consider using more training "
                "data or reducing --vocab-size to match your corpus size.",
                requested, vocab_size_local, min_vocab_size, max_vocab_size
            )
        spm.SentencePieceTrainer.Train(
            input=tmp_corpus,
            model_prefix=model_prefix,
            vocab_size=vocab_size_local,
            character_coverage=0.9995,
            model_type="unigram", # or 'bpe'
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )
        model_file = model_prefix + ".model"
        vocab_file = model_prefix + ".vocab"
        logger.info("Trained SentencePiece model: %s (vocab_size=%d)", model_file, vocab_size_local)
        os.remove(tmp_corpus) # corpus file cleanup
        return model_file, vocab_file

    @staticmethod
    def load_sp_processor(model_path: str):
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        return sp

    @staticmethod
    def encode_with_sp(sp, texts: List[str]) -> List[List[int]]:
        return [sp.EncodeAsIds(s) for s in texts]

    @staticmethod
    def decode_with_sp(sp, ids: List[int]) -> str:
        return sp.DecodeIds(ids)

    @staticmethod
    def build_keras_tokenizer(texts: List[str], num_words: int):
        """fallback simple word-level tokenizer (Keras)"""
        tok = keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<unk>", filters="")
        tok.fit_on_texts(texts)
        return tok

    @staticmethod
    def texts_to_sequences_keras(tok, texts: List[str]) -> List[List[int]]:
        return tok.texts_to_sequences(texts)

    def prepareTokenData(self, input_texts: List[str], target_texts: List[str],
                           direction: str, vocab_size_local: int, max_samples: Optional[int] = None) -> Dict:
        """
        tokenized and padded arrays for encoder/decoder training using SentencePiece
        or Keras tokenizer fallback are preped, returns dict with arrays and tokenizers/meta
        """
        logger.info("Preparing token data (samples=%s, vocab_size=%d)", str(max_samples), vocab_size_local)
        if max_samples:
            input_texts = input_texts[:max_samples]
            target_texts = target_texts[:max_samples]

        # sp training on combined corpus i.e. (source + target) 
        combined = input_texts + target_texts
        sp_model_prefix = str(self.trained_models_dir / f"spm_{direction}")
        if self.use_sentencepiece and HAVE_SPM:
            sp_model, sp_vocab = self.trainSentencePiece(combined, sp_model_prefix, vocab_size_local)
            sp = self.load_sp_processor(sp_model)
            # encode
            encoder_seq = self.encode_with_sp(sp, input_texts)
            decoder_seq = self.encode_with_sp(sp, target_texts)
            token_tool = ("sentencepiece", sp_model) # store model path
            logger.info("SentencePiece used. Model at: %s", sp_model)
        else:
            # fallback Keras tokenizer on words
            logger.warning("sentencepiece not available - falling back to Keras word-level tokenizer.")
            tok = self.build_keras_tokenizer(combined, vocab_size_local)
            encoder_seq = self.texts_to_sequences_keras(tok, input_texts)
            decoder_seq = self.texts_to_sequences_keras(tok, target_texts)
            token_tool = ("keras_tokenizer", tok) # serialized below as json str
            logger.info("Keras tokenizer built with vocab size approx: %d", min(vocab_size_local, len(tok.word_index) + 1))

        # add START and END tokens as special tokens in target sequences
        # for sp we used BOS/EOS reserved ids (2 and 3) while sp.encode_as_ids doesn't automatically add BOS/EOS
        # therefore we will explicitly add BOS(2) and EOS(3) if using sentencepiece. For Keras tokenizer we'll use reserved indices
        if self.use_sentencepiece and HAVE_SPM:
            BOS_ID = 2
            EOS_ID = 3
            encoder_input_ids = encoder_seq
            decoder_input_ids = [[BOS_ID] + s for s in decoder_seq]
            decoder_target_ids = [s + [EOS_ID] for s in decoder_seq]
            input_vocab_size = vocab_size_local
            target_vocab_size = vocab_size_local
        else:
            # with Keras tokenizer, ensure special tokens exist:
            tok = token_tool[1]
            # reserve indices: 1 -> oov by Keras default. We'll create tokens for <bos> and <eos>
            # append to tokenizer.word_index if not present (note: Keras Tokenizer is not trivial to extend, instead we will map manually)
            BOS = "<bos>"
            EOS = "<eos>"
            if BOS not in tok.word_index:
                tok.word_index[BOS] = len(tok.word_index) + 1
            if EOS not in tok.word_index:
                tok.word_index[EOS] = len(tok.word_index) + 1
            bos_id = tok.word_index[BOS]
            eos_id = tok.word_index[EOS]
            encoder_input_ids = encoder_seq
            decoder_input_ids = [[bos_id] + s for s in decoder_seq]
            decoder_target_ids = [s + [eos_id] for s in decoder_seq]
            input_vocab_size = min(vocab_size_local, len(tok.word_index) + 1)
            target_vocab_size = input_vocab_size
            # Keras tokenizer serialized as a JSON string instead of storing the raw object coz Raw Tokenizer objects are not reliably picklable across Keras ver.(s), 
            # & the class deprecated in newer Keras, the type tag changes to "keras_tokenizer_json" so loaders can detect format
            token_tool = ("keras_tokenizer_json", tok.to_json())
 
        max_enc_len = max(len(s) for s in encoder_input_ids)
        max_dec_len = max(len(s) for s in decoder_input_ids)

        # post padding sequences
        encoder_input_padded = keras.preprocessing.sequence.pad_sequences(encoder_input_ids, maxlen=max_enc_len, padding="post")
        decoder_input_padded = keras.preprocessing.sequence.pad_sequences(decoder_input_ids, maxlen=max_dec_len, padding="post")
        decoder_target_padded = keras.preprocessing.sequence.pad_sequences(decoder_target_ids, maxlen=max_dec_len, padding="post")

        logger.info("Prepared arrays shapes: encoder=%s, decoder_input=%s, decoder_target=%s",
                    str(encoder_input_padded.shape), str(decoder_input_padded.shape), str(decoder_target_padded.shape))

        return {
            "encoder_input": encoder_input_padded,
            "decoder_input": decoder_input_padded,
            "decoder_target": decoder_target_padded,
            "max_enc_len": max_enc_len,
            "max_dec_len": max_dec_len,
            "input_vocab_size": input_vocab_size,
            "target_vocab_size": target_vocab_size,
            "token_tool": token_tool
        }

    def encodeOnly(self, input_texts: List[str], target_texts: List[str],
                   token_tool: Tuple, max_enc_len: int, max_dec_len: int) -> Dict:
        """
        encode texts using an already-trained token_tool without re-training the tokenizer,
        Used in two places:
          1. Encoding the test split after training (tokenizer was trained on train split only) &
          2. eval_only mode, where the saved token_tool is loaded from disk rather than retraining
        Handles both sentencepiece ("sentencepiece", model_path) and the serialized Keras fallback
        ("keras_tokenizer_json", json_str) as well as the legacy ("keras_tokenizer", obj) format
        """
        tool_type, tool_obj = token_tool
        if tool_type == "sentencepiece" and HAVE_SPM:
            sp = self.load_sp_processor(tool_obj)
            encoder_seq = self.encode_with_sp(sp, input_texts)
            decoder_seq = self.encode_with_sp(sp, target_texts)
            BOS_ID = 2
            EOS_ID = 3
        else:
            # resolve tokenizer object from whichever serialization format was used
            if tool_type == "keras_tokenizer_json":
                try:
                    tok = keras.preprocessing.text.tokenizer_from_json(tool_obj)
                except AttributeError:
                    logger.warning("tokenizer_from_json not available: falling back to raw object")
                    tok = tool_obj
            else:
                tok = tool_obj # legacy, raw Keras Tokenizer obj.
            encoder_seq = self.texts_to_sequences_keras(tok, input_texts)
            decoder_seq = self.texts_to_sequences_keras(tok, target_texts)
            BOS_ID = tok.word_index.get("<bos>") or tok.word_index.get("bos") or 1
            EOS_ID = tok.word_index.get("<eos>") or tok.word_index.get("eos") or 1

        encoder_input_ids = encoder_seq
        decoder_input_ids = [[BOS_ID] + s for s in decoder_seq]
        decoder_target_ids = [s + [EOS_ID] for s in decoder_seq]

        encoder_input_padded = keras.preprocessing.sequence.pad_sequences(
            encoder_input_ids, maxlen=max_enc_len, padding="post")
        decoder_input_padded = keras.preprocessing.sequence.pad_sequences(
            decoder_input_ids, maxlen=max_dec_len, padding="post")
        decoder_target_padded = keras.preprocessing.sequence.pad_sequences(
            decoder_target_ids, maxlen=max_dec_len, padding="post")

        logger.info("encodeOnly shapes: encoder=%s decoder_input=%s decoder_target=%s",
                    str(encoder_input_padded.shape), str(decoder_input_padded.shape), str(decoder_target_padded.shape))
        return {
            "encoder_input": encoder_input_padded,
            "decoder_input": decoder_input_padded,
            "decoder_target": decoder_target_padded,
        }


class Seq2SeqModel:
    """
    subword token-level seq2seq model w/ attention 
    """

    def __init__(self, trained_models_dir: Path):
        self.trained_models_dir = Path(trained_models_dir)

    @staticmethod
    def build_token_seq2seq(input_vocab_size: int, target_vocab_size: int,
                            max_enc_len: Optional[int] = None, max_dec_len: Optional[int] = None,
                            embedding_dim_local: int = 256, latent_dim_local: int = 512,
                            dropout_local: float = 0.3, recurrent_dropout_local: float = 0.0) -> Tuple[keras.Model, dict]:
        """builds token-level seq2seq model with attention, returns: (training_model, component_layers_dict)
        component_layers_dict contains pointers to layers needed for inference i.e. (embedding layers, encoder_bidirectional_layer, decoder_lstm_layer, attention_layer, decoder_dense_layer)"""
        logger.info("Building token-level seq2seq model: input_vocab=%d target_vocab=%d emb=%d latent=%d",
                    input_vocab_size, target_vocab_size, embedding_dim_local, latent_dim_local)
        # encoder
        encoder_inputs = keras.Input(shape=(None,), name="encoder_inputs") # token ids
        enc_emb = keras.layers.Embedding(input_vocab_size, embedding_dim_local, mask_zero=True, name="encoder_embedding")(encoder_inputs)

        # BiLSTM encoder, return_sequences=True so attention later can attend over encoder outputs
        encoder_bi = keras.layers.Bidirectional(
            keras.layers.LSTM(latent_dim_local, return_sequences=True, return_state=True,
                              dropout=dropout_local, recurrent_dropout=recurrent_dropout_local),
            merge_mode="concat",
            name="encoder_bilstm"
        )
        
        # encoder_bi(enc_emb): (output, f_h, f_c, b_h, b_c)
        encoder_outputs, f_h, f_c, b_h, b_c = encoder_bi(enc_emb)

        # concat states for decoder init
        state_h = keras.layers.Concatenate(name="encoder_h_concat")([f_h, b_h])
        state_c = keras.layers.Concatenate(name="encoder_c_concat")([f_c, b_c])
        encoder_states = [state_h, state_c]

        # decoder
        decoder_inputs = keras.Input(shape=(None,), name="decoder_inputs") # token ids
        dec_emb = keras.layers.Embedding(target_vocab_size, embedding_dim_local, mask_zero=True, name="decoder_embedding")(decoder_inputs)

        decoder_lstm_units = latent_dim_local * 2 # match concatenated encoder states
        decoder_lstm = keras.layers.LSTM(decoder_lstm_units, return_sequences=True, return_state=True,
                                         dropout=dropout_local, recurrent_dropout=recurrent_dropout_local, name="decoder_lstm")
        decoder_outputs_and_states = decoder_lstm(dec_emb, initial_state=encoder_states)
        decoder_outputs = decoder_outputs_and_states[0]
        # attention: use Keras Attention layer (dot-product attention). Provide decoder_outputs as query and encoder_outputs as value
        attention_layer = keras.layers.Attention(name="attention_layer")
        context = attention_layer([decoder_outputs, encoder_outputs]) # shape (batch, dec_steps, enc_dim)
        # concat context and decoder_outputs
        decoder_concat = keras.layers.Concatenate(axis=-1, name="decoder_concat")([decoder_outputs, context])
        decoder_dense = keras.layers.Dense(target_vocab_size, activation="softmax", name="decoder_dense")
        decoder_outputs_final = decoder_dense(decoder_concat)

        # finales model
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs_final, name="seq2seq_token_model")
        logger.info("Model built successfully")
        model.summary(print_fn=logger.info)
        components = {
            "encoder_inputs": encoder_inputs,
            "encoder_outputs": encoder_outputs,
            "encoder_states": encoder_states,
            "decoder_inputs": decoder_inputs,
            "decoder_embedding": model.get_layer("decoder_embedding"),
            "encoder_embedding": model.get_layer("encoder_embedding"),
            "decoder_lstm": decoder_lstm,
            "attention_layer": attention_layer,
            "decoder_dense": decoder_dense
        }
        return model, components

    @staticmethod
    def build_inference_models_from_trained(trained_model_path: str, latent_dim_local: int, direction_suffix: str):
        """given a fully trained model saved to trained_model_path, we just construct:
          - encoder_model: inputs -> [encoder_outputs, state_h, state_c] (we need encoder_outputs for attention)
          - decoder_model: [prev_token, prev_h, prev_c, encoder_outputs] -> [probs, next_h, next_c]"""
        logger.info("Loading full model from %s to build inference models", trained_model_path)
        full = keras.models.load_model(trained_model_path, compile=False)
        # locate layers by name to reuse weights
        try:
            encoder_embedding = full.get_layer("encoder_embedding")
            encoder_bilstm = full.get_layer("encoder_bilstm")
            decoder_embedding = full.get_layer("decoder_embedding")
            decoder_lstm_layer = full.get_layer("decoder_lstm")
            attention_layer = full.get_layer("attention_layer")
            decoder_dense = full.get_layer("decoder_dense")
        except Exception as e:
            logger.exception("Failed to find named layers in the saved model. Layer names: %s", [l.name for l in full.layers])
            raise e

        # rebuild encoder model: takes encoder_inputs -> encoder_outputs, state_h, state_c (concatenated)
        encoder_inputs_inf = keras.Input(shape=(None,), name="encoder_inputs_inf")
        enc_emb_inf = encoder_embedding(encoder_inputs_inf)
        encoder_outputs_inf, f_h_inf, f_c_inf, b_h_inf, b_c_inf = encoder_bilstm(enc_emb_inf)
        # named Concatenate layers (have no trained weights, but named for debuggability)
        state_h_inf = keras.layers.Concatenate(name="encoder_h_concat_inf")([f_h_inf, b_h_inf])
        state_c_inf = keras.layers.Concatenate(name="encoder_c_concat_inf")([f_c_inf, b_c_inf])
        encoder_model = keras.Model(
            encoder_inputs_inf,
            [encoder_outputs_inf, state_h_inf, state_c_inf],
            name=f"encoder_infer_{direction_suffix}"
        )

        # decoder inference: Inputs: prev_token (1,), prev_h, prev_c, encoder_outputs
        decoder_token_input = keras.Input(shape=(1,), name="decoder_token_input") # single step input
        prev_h = keras.Input(shape=(latent_dim_local * 2,), name="decoder_prev_h")
        prev_c = keras.Input(shape=(latent_dim_local * 2,), name="decoder_prev_c")
        encoder_out_input = keras.Input(shape=(None, latent_dim_local * 2), name="encoder_out_input")

        # reuse decoder_embedding & decoder_lstm_layer & attention_layer & decoder_dense
        dec_emb_step = decoder_embedding(decoder_token_input) # shape (batch, 1, emb)
        # run one step LSTM with initial state = [prev_h, prev_c]
        dec_outputs_step, next_h, next_c = decoder_lstm_layer(dec_emb_step, initial_state=[prev_h, prev_c])
        # attention between this decoder step outputs (query with length=1) and full encoder_out
        context_step = attention_layer([dec_outputs_step, encoder_out_input])
        concat_step = keras.layers.Concatenate(axis=-1)([dec_outputs_step, context_step])
        output_step = decoder_dense(concat_step) # shape (batch, 1, target_vocab)
        decoder_model = keras.Model([decoder_token_input, prev_h, prev_c, encoder_out_input], [output_step, next_h, next_c], name=f"decoder_infer_{direction_suffix}")

        logger.info("Building inference encoder and decoder models done")
        return encoder_model, decoder_model

    @staticmethod
    def compile_and_train(model, encoder_input, decoder_input, decoder_target,
                          save_prefix: str, batch_size_local: int, epochs_local: int,
                          initial_lr: float = 1e-6, # compiled @ warmup lr 
                          callbacks=None):
        """compile and train the token-level model. decoder_target is integer token ids (padded). We r using sparse_categorical_crossentropy so targets are ints (not one-hot)"""
        logger.info("Compiling model and preparing dataset") 
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_lr, clipnorm=1.0),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=["sparse_categorical_accuracy"])
        
        # a per-timestep sample_weight mask so pad tokens (id=0) do not contribute to the loss. 
        # Without this the model was wasting capacity learning to predict pads, thereby inflating the loss signal and degrading translation quality
        sample_weight = (decoder_target != 0).astype(np.float32) # shape (N, dec_len), 0.0 at pad positions
        logger.info("Starting fit: batch=%d epochs=%d", batch_size_local, epochs_local)
        history = model.fit([encoder_input, decoder_input],
                            np.expand_dims(decoder_target, -1), # shape (batch, dec_steps, 1) for sparse loss
                            sample_weight=sample_weight, # Pad Masking, exclude pad positions from loss
                            batch_size=batch_size_local,
                            epochs=epochs_local,
                            validation_split=0.2,
                            callbacks=callbacks or [])
        # save model
        save_path = str(Path(trained_models_dir) / f"sq2sq_model_{save_prefix}.keras")
        model.save(save_path, overwrite=True)
        logger.info("Saved trained model to %s", save_path)
        return history, save_path

    @staticmethod
    def greedy_decode_sequence(encoder_model, decoder_model, input_seq, token_tool, max_dec_len,
                               direction, use_sp=True,
                               sp_processor=None # to accept pre-loaded processor
                               ) -> str:
        """greedy stepwise decoding with the inference encoder_model & decoder_model, returns decoded string (tokens to text)
        token_tool is ("sentencepiece", model_path) or ("keras_tokenizer_json", json_str) or legacy ("keras_tokenizer", obj)
        sp_processor may be a pre-loaded SentencePieceProcessor so the model
        file is not re-opened from disk on every call. evaluate_on_testset always supplies this"""
        # encoder_model -> encoder_outputs, state_h, state_c
        # input_seq should be shape (seq_len,) or (1, seq_len)
        if isinstance(input_seq, np.ndarray) and input_seq.ndim == 2:
            enc_input_for_pred = input_seq
            encoder_outputs, h, c = encoder_model.predict(enc_input_for_pred, verbose=0)
        else:
            encoder_outputs, h, c = encoder_model.predict(np.array([input_seq]), verbose=0)

        # prepare first decoder token: BOS
        if use_sp and HAVE_SPM:
            BOS_ID = 2
            EOS_ID = 3
        else:
            # resolved Keras tokenizer from either serialization format
            tool_type, tool_obj = token_tool
            if tool_type == "keras_tokenizer_json":
                try:
                    tok = keras.preprocessing.text.tokenizer_from_json(tool_obj)
                except AttributeError:
                    tok = tool_obj
            else:
                tok = tool_obj # then legacy, raw tokenizer object
            bos_id = tok.word_index.get("<bos>") or tok.word_index.get("bos") or 1
            eos_id = tok.word_index.get("<eos>") or tok.word_index.get("eos") or 1
            BOS_ID = bos_id
            EOS_ID = eos_id

        decoded_ids: List[int] = []
        current_token = np.array([[BOS_ID]]) # shape (1,1)
        next_h = h
        next_c = c
        for _ in range(max_dec_len):
            preds, next_h, next_c = decoder_model.predict([current_token, next_h, next_c, encoder_outputs], verbose=0)
            # preds shape (1,1,vocab)
            preds = preds[0, -1, :] # (vocab,)
            idx = int(np.argmax(preds))
            if idx == EOS_ID:
                break
            decoded_ids.append(idx)
            current_token = np.array([[idx]])
        # decode token ids back to text
        if use_sp and HAVE_SPM:
            # use pre-loaded processor if available,
            # fall back to loading from disk only when no processor was passed in (z.B. in standalone/ad-hoc calls)
            if sp_processor is not None:
                sp = sp_processor
            else:
                sp = spm.SentencePieceProcessor()
                sp.Load(token_tool[1]) # token_tool = ("sentencepiece", model_path)
            text = sp.DecodeIds(decoded_ids)
        else:
            # tok was resolved above from either JSON or legacy format
            # inverse map
            inv = {v: k for k, v in tok.word_index.items()}
            words = [inv.get(i, "") for i in decoded_ids]
            text = " ".join(w for w in words if w)
        return text

    @staticmethod
    def evaluate_on_testset(encoder_model, decoder_model, encoder_inputs, input_texts, target_texts, token_tool, max_enc_len, max_dec_len, direction, sample_limit=None):
        """greedy decoding on held-out test set and compute CER, exact-match etc.
        encoder_inputs: padded integer arrays (N, enc_len)
        input_texts, target_texts: original text lists (for decoding references)
        token_tool: ("sentencepiece", model_path) or ("keras_tokenizer_json", json_str)
        """
        logger.info("Starting evaluation: N=%d", encoder_inputs.shape[0])
        sp_proc = None
        if token_tool[0] == "sentencepiece" and HAVE_SPM:
            sp_proc = spm.SentencePieceProcessor()
            sp_proc.Load(token_tool[1])
            logger.info("Pre-loaded SentencePiece processor for evaluation (avoided per-call to disk I/O) :)")
        results: List[Dict] = []
        N = encoder_inputs.shape[0] if sample_limit is None else min(sample_limit, encoder_inputs.shape[0])
        for i in range(N):
            enc_seq = encoder_inputs[i]
            src = input_texts[i]
            ref = target_texts[i].strip()
            pred_text = Seq2SeqModel.greedy_decode_sequence(
                encoder_model, decoder_model, enc_seq, token_tool, max_dec_len, direction,
                use_sp=(token_tool[0] == "sentencepiece"),
                sp_processor=sp_proc # pre-loaded processor 
            )
            results.append({"index": i, "source": src, "reference": ref, "prediction": pred_text})
            if i % 200 == 0 and i > 0:
                logger.info("Decoded %d / %d", i, N)
        # compute metrics
        avg_cer = float(np.mean([Evaluator.cer(r["reference"], r["prediction"]) for r in results])) if results else 0.0
        exact = sum(1 for r in results if r["reference"] == r["prediction"])
        exact_acc = exact / len(results) if results else 0.0
        metrics = {
            "direction": direction,
            "average_CER": avg_cer,
            "exact_match_accuracy": exact_acc,
            "num_test_samples": len(results),
            "avg_pred_length": float(np.mean([len(r["prediction"]) for r in results])) if results else 0.0,
            "avg_ref_length": float(np.mean([len(r["reference"]) for r in results])) if results else 0.0,
            "timestamp": datetime.datetime.now().isoformat()
        }
        return metrics, results



def run_pipeline(direction: str = "eng2deu", # erfoderlich
                 data_path: str = data_path_default,
                 samples: int = set_num_samples,
                 vocab_size_local: int = vocab_size,
                 train_only: bool = False,
                 eval_only: bool = False,
                 # optionals
                 max_eval_samples: Optional[int] = None,
                 epochs_local: Optional[int] = None,
                 batch_size_local: Optional[int] = None,
                 embedding_dim_local: Optional[int] = None,
                 latent_dim_local: Optional[int] = None,
                 dropout_local: Optional[float] = None,
                 recurrent_dropout_local: Optional[float] = None,
                 peak_lr_local: Optional[float] = None,
                 warmup_initial_lr_local: Optional[float] = None,
                 warmup_epochs_local: Optional[int] = None):
    """ohhmaa gawd paramsss °~°"""
    # default behaviour kept incase no explicit hyparams provided: None params to current module-level globals,  
    # & natürlich full CLI override when called from __main__
    if epochs_local is None: epochs_local = epochs
    if batch_size_local is None: batch_size_local = batch_size
    if embedding_dim_local is None: embedding_dim_local = embedding_dim
    if latent_dim_local is None: latent_dim_local = latent_dim
    if dropout_local is None: dropout_local = dropout
    if recurrent_dropout_local is None: recurrent_dropout_local = recurrent_dropout
    if peak_lr_local is None: peak_lr_local = peak_lr
    if warmup_initial_lr_local is None: warmup_initial_lr_local = warmup_initial_lr
    if warmup_epochs_local is None: warmup_epochs_local = warmup_epochs

    assert direction in ("deu2eng", "eng2deu")
    logger.info("%s", "-" * 60)
    logger.info("STARTING SEQ2SEQ PIPELINE (direction=%s)", direction)
    logger.info("%s", "-" * 60)

    # Prepare data
    logger.info("Step 0: based on direction, read raw data, produce input_texts and target_texts")
    with open(data_path, encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().split("\n") if ln.strip()]
    logger.info("Total lines in file: %d", len(lines))

    input_texts: List[str] = []
    target_texts: List[str] = []
    for line in lines[:samples]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        if direction == "eng2deu":
            src, tgt = parts[0], parts[1]
        else:
            tgt, src = parts[0], parts[1]
        # training format expects target with no extra manual BOS/EOS, we'll add tokens later
        input_texts.append(src)
        target_texts.append(tgt)

    logger.info("Read %d examples, (direction=%s)", len(input_texts), direction)

    n = min(len(input_texts), samples)
    train_idx_raw, test_idx_raw = Evaluator.train_test_split_indices(n, test_ratio=0.2, seed=42)
    input_texts_train_raw = [input_texts[i] for i in train_idx_raw]
    target_texts_train_raw = [target_texts[i] for i in train_idx_raw]
    input_texts_test = [input_texts[i] for i in test_idx_raw]
    target_texts_test = [target_texts[i] for i in test_idx_raw]
    logger.info("Raw-text split (before tokenization): train=%d test=%d",
                len(input_texts_train_raw), len(input_texts_test))

    preprocessor = DataPreprocessor(trained_models_dir=trained_models_dir, use_sp=use_sentencepiece, vocab_size_default=vocab_size_local)

    # paths defined early so both training and eval_only branches can reference them
    sq2sq_model_path = trained_models_dir / f"sq2sq_model_{direction}.keras"
    encoder_model_path = trained_models_dir / f"encoder_model_{direction}.keras"
    decoder_model_path = trained_models_dir / f"decoder_model_{direction}.keras"
    token_info_file = trained_models_dir / f"token_tool_{direction}.pkl"

    # now we simply load the persisted token_tool pickle & go straight to test-set encoding
    if eval_only:
        if not encoder_model_path.exists() or not decoder_model_path.exists():
            logger.error("Inference models not found. Please run training first.")
            return
        if not token_info_file.exists():
            logger.error("Token info file %s not found. Please run training first.", token_info_file)
            return
        with open(token_info_file, "rb") as f:
            token_tool = pickle.load(f)
            max_enc_len = pickle.load(f)
            max_dec_len = pickle.load(f)
            input_vocab_size = pickle.load(f)
            target_vocab_size = pickle.load(f)
        logger.info("eval_only: loaded token_tool from %s (skipping SPM retraining)", token_info_file)
        # no retraining, we encode test texts with the saved tokenizer  
        test_encoded = preprocessor.encodeOnly(input_texts_test, target_texts_test, token_tool, max_enc_len, max_dec_len)
        encoder_test = test_encoded["encoder_input"]
        encoder_model = keras.models.load_model(str(encoder_model_path), compile=False)
        decoder_model = keras.models.load_model(str(decoder_model_path), compile=False)
        metrics, results = Seq2SeqModel.evaluate_on_testset(
            encoder_model, decoder_model, encoder_test, input_texts_test, target_texts_test,
            token_tool, max_enc_len, max_dec_len, direction, sample_limit=max_eval_samples)
        # results saved
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = experiments_dir / f"eval_{direction}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        Evaluator.save_predictions(results, str(exp_dir / "predictions.jsonl"))
        Evaluator.save_metrics(metrics, str(exp_dir / "metrics.json"))
        logger.info("Eval completed. Metrics: %s", json.dumps(metrics, indent=2))
        return {"metrics": metrics, "predictions": str(exp_dir / "predictions.jsonl")}

    # token data prep - training data only (no SPM data leakage into test set)
    token_data = preprocessor.prepareTokenData(input_texts_train_raw, target_texts_train_raw, direction, vocab_size_local)
    encoder_train = token_data["encoder_input"]
    decoder_input_train = token_data["decoder_input"]
    decoder_target_train = token_data["decoder_target"]
    max_enc_len = token_data["max_enc_len"]
    max_dec_len = token_data["max_dec_len"]
    input_vocab_size = token_data["input_vocab_size"]
    target_vocab_size = token_data["target_vocab_size"]
    token_tool = token_data["token_tool"]

    # token tool info. saved for inference later
    with open(token_info_file, "wb") as f:
        pickle.dump(token_tool, f)
        pickle.dump(max_enc_len, f)
        pickle.dump(max_dec_len, f)
        pickle.dump(input_vocab_size, f)
        pickle.dump(target_vocab_size, f)
    logger.info("Saved token tool info to %s", str(token_info_file)) 

    # encode the test partition using the already-trained tokenizer - no retraining.
    test_encoded = preprocessor.encodeOnly(input_texts_test, target_texts_test, token_tool, max_enc_len, max_dec_len)
    encoder_test = test_encoded["encoder_input"]
    logger.info("Encoded test set: encoder_test=%s", str(encoder_test.shape))
    logger.info("Split: train=%d test=%d", encoder_train.shape[0], encoder_test.shape[0])

    # build model
    model, components = Seq2SeqModel.build_token_seq2seq(input_vocab_size, target_vocab_size,
                                                        embedding_dim_local=embedding_dim_local, 
                                                        latent_dim_local=latent_dim_local,
                                                        dropout_local=dropout_local,
                                                        recurrent_dropout_local=recurrent_dropout_local)

    # callbacks: early stopping + reduceLR + tensorboard
    fit_callbacks = []
    tb_logdir = log_path / ("fit_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    fit_callbacks.append(keras.callbacks.TensorBoard(log_dir=str(tb_logdir), histogram_freq=1))
    fit_callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True))
    fit_callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1))
    # warmup callback before ReduceLROnPlateau, it fires on_epoch_begin,
    # overriding whatever RLROP set at the previous epoch-end for the duration of warmup_epochs
    fit_callbacks.append(WarmUpLRCallback(
        warmup_epochs=warmup_epochs_local,
        peak_lr=peak_lr_local,
        warmup_lr=warmup_initial_lr_local
    ))

    # train
    history, save_path = Seq2SeqModel.compile_and_train(
        model, encoder_train, decoder_input_train, decoder_target_train,
        save_prefix=direction,
        batch_size_local=batch_size_local, 
        epochs_local=epochs_local,
        initial_lr=warmup_initial_lr_local, # compiling @ warmup-starting lr (1e-6)
        callbacks=fit_callbacks)

    # from full model we build inference models..
    encoder_model, decoder_model = Seq2SeqModel.build_inference_models_from_trained(
        save_path, latent_dim_local, direction)
    encoder_model.save(str(encoder_model_path), overwrite=True)
    decoder_model.save(str(decoder_model_path), overwrite=True)
    logger.info("Saved inference models: %s, %s", str(encoder_model_path), str(decoder_model_path))

    # token_tool save (already saved earlier) and also history
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"token_seq2seq_{direction}_{timestamp}"
    exp_dir = experiments_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "train_history.json", "w", encoding="utf-8") as fh:
        json.dump(history.history, fh, indent=2)

    # if we're doin train_only, skip the long eval
    if train_only:
        logger.info("Train-only requested: skipping evaluation.")
        return {"history": history.history, "model_path": save_path}

    # eval. on held-out test set
    metrics, results = Seq2SeqModel.evaluate_on_testset(
        encoder_model, decoder_model, encoder_test, input_texts_test, target_texts_test,
        token_tool, max_enc_len, max_dec_len, direction, sample_limit=max_eval_samples)

    # predictions & metrics saved
    Evaluator.save_predictions(results, str(exp_dir / "predictions.jsonl"))
    Evaluator.save_metrics(metrics, str(exp_dir / "metrics.json"))
    logger.info("Evaluation metrics: %s", json.dumps(metrics, indent=2))
    return {"history": history.history, "metrics": metrics, "predictions": str(exp_dir / "predictions.jsonl")}


# cli
def parse_args():
    # all tunable hyperparams are overrideable
    p = argparse.ArgumentParser(description="Train/eval token-level seq2seq with SentencePiece + attention.")
    p.add_argument("--direction", choices=["deu2eng", "eng2deu"], default="eng2deu")
    p.add_argument("--data-path", default=data_path_default)
    p.add_argument("--samples", type=int, default=set_num_samples)
    p.add_argument("--vocab-size", type=int, default=vocab_size)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--embedding-dim", type=int, default=embedding_dim)
    p.add_argument("--latent-dim", type=int, default=latent_dim) 
    p.add_argument("--dropout", type=float, default=dropout)
    p.add_argument("--recurrent-dropout", type=float, default=recurrent_dropout,
                   help="Set >0 to regularize: disables CuDNN LSTM (slower on GPU).")
    p.add_argument("--peak-lr", type=float, default=peak_lr,
                   help="Peak learning rate reached at end of warmup (default: 1e-3).")
    p.add_argument("--warmup-epochs", type=int, default=warmup_epochs,
                   help="Number of epochs for linear LR warmup before ReduceLROnPlateau takes over.")
    p.add_argument("--train-only", action="store_true", help="Train only, skip evaluation (faster trains).")
    p.add_argument("--eval-only", action="store_true", help="Only evaluate using saved inference models (no training).")
    p.add_argument("--max-eval-samples", type=int, default=None, help="Limit number of eval samples (for faster eval).")
    p.add_argument("--no-sentencepiece", action="store_true", help="Disable sentencepiece and use Keras tokenizer fallback.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # cli overrides to module-level defaults where appropriate
    if args.no_sentencepiece:
        use_sentencepiece = False
        logger.info("SentencePiece disabled by default, using word-level tokenizer fallback.")
    epochs = args.epochs
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim
    latent_dim = args.latent_dim
    set_num_samples = args.samples
    vocab_size = args.vocab_size 
    
    if args.eval_only:
        result = run_pipeline(
            direction=args.direction, data_path=args.data_path, samples=args.samples,
            vocab_size_local=args.vocab_size, train_only=False, eval_only=True,
            max_eval_samples=args.max_eval_samples,
            epochs_local=args.epochs, batch_size_local=args.batch_size,
            embedding_dim_local=args.embedding_dim, latent_dim_local=args.latent_dim,
            dropout_local=args.dropout, recurrent_dropout_local=args.recurrent_dropout,
            peak_lr_local=args.peak_lr, warmup_epochs_local=args.warmup_epochs,
            warmup_initial_lr_local=warmup_initial_lr # not exposed as CLI arg, usin module default
        )
    else:
        result = run_pipeline(
            direction=args.direction, data_path=args.data_path, samples=args.samples,
            vocab_size_local=args.vocab_size, train_only=args.train_only, eval_only=False,
            max_eval_samples=args.max_eval_samples,
            epochs_local=args.epochs, batch_size_local=args.batch_size,
            embedding_dim_local=args.embedding_dim, latent_dim_local=args.latent_dim,
            dropout_local=args.dropout, recurrent_dropout_local=args.recurrent_dropout,
            peak_lr_local=args.peak_lr, warmup_epochs_local=args.warmup_epochs,
            warmup_initial_lr_local=warmup_initial_lr # not exposed as CLI arg, usin module default
        )

    logger.info("Done. -_-! Result: %s", str(result))
