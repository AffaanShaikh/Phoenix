"""
Inference CLI for the char-level seq2seq models.
Usage examples:
    Single-shot:
        python infer_translate.py -s "how are you?" --direction en-de \
            --char2enc r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\char2encoding.pkl" \
            --encoder r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\encoder_model.keras" \
            --decoder r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\decoder_model.keras"

    Interactive:
        python infer_translate.py --interactive --direction en-de

Notes:
- To translate the other way (de-en) you must have a separate trained
  encoder/decoder pair and a matching char2encoding file for that direction.
- char2encoding file must match the models (char->index and index->char).
- The script will try to handle missing characters (warn & skip them).

Added CLI flags:
- --sample-eval runs the beam decoder on a small built-in list of test sentences.
- --test-file path expects a TSV file src<TAB>ref and will run batch eval and print/save metrics.
- --auto-build-inference will try to build encoder/decoder inference models from sq2sq_model.keras
  if encoder_model.keras or decoder_model.keras are missing.
- --beam-width controls beam search width (use 1 for greedy).

"""

import argparse
import json
import os
import sys
import logging
import heapq
import datetime
try:
    from tensorflow import keras
except Exception:
    import keras
import numpy as np
import pickle 

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("infer_translate")

# Default paths for my local setup; you can override with CLI flags
DEFAULT_CHAR2ENC = r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\char2encoding_eng2deu.pkl"
DEFAULT_ENCODER = r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\encoder_model_eng2deu.keras"
DEFAULT_DECODER = r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\decoder_model_eng2deu.keras"
DEFAULT_SQ2SQ_MODEL = os.path.join(os.path.dirname(DEFAULT_ENCODER), "sq2sq_model_eng2deu.keras")


# helpers for (de)serialization and encoding
def load_char2encoding(path):
    """
    Loads the char2encoding pickle produced by your training pipeline.
    The training pipeline stored, in order:
      - input_token_index (dict: char -> int)
      - max_encoder_seq_length (int)
      - num_encoder_tokens (int)
      - reverse_target_char_index (dict: int -> char)
      - num_decoder_tokens (int)
      - target_token_index (dict: char -> int)

    Returns these items in the same order.
    """
    with open(path, "rb") as f:
        input_token_index = pickle.load(f)
        max_encoder_seq_length = pickle.load(f)
        num_encoder_tokens = pickle.load(f)
        reverse_target_char_index = pickle.load(f)
        num_decoder_tokens = pickle.load(f)
        target_token_index = pickle.load(f)
    return input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index


def encoding_sentence_to_predict(sentence, input_token_index, max_encoder_seq_length, num_encoder_tokens):
    """
    Create one-hot encoder input of shape (1, max_encoder_seq_length, num_encoder_tokens).
    Characters not in input_token_index are skipped and a warning is logged.
    """
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    for t, ch in enumerate(sentence):
        if t >= max_encoder_seq_length:
            break
        if ch in input_token_index:
            encoder_input_data[0, t, input_token_index[ch]] = 1.0
        else:
            # commonly we might prefer to map unknown -> space or an UNK token,
            # but current pipeline expects a space ' ' padding; we skip and warn.
            logger.warning("Character '%s' not in input_token_index; skipping it", ch)
    # pad remainder with space token if available
    if " " in input_token_index:
        if len(sentence) < max_encoder_seq_length:
            encoder_input_data[0, len(sentence):, input_token_index[" "]] = 1.0
    return encoder_input_data


def decode_sequence_inference(input_seq, encoder_model, decoder_model, num_decoder_tokens, target_token_index,
                              reverse_target_char_index, max_decoder_seq_length_given=100):
    """
    Autoregressive (greedy) decoding loop. Mirrors the training/inference splitting in your pipeline.
    - input_seq: output from encoding_sentence_to_predict
    - encoder_model, decoder_model: saved inference models (encoder outputs states, decoder accepts previous token + states)
    - target_token_index: dict char -> idx (used to start sequence with '\t')
    - reverse_target_char_index: dict idx -> char (used to map argmax back to char)
    - max_decoder_seq_length_given: safety cutoff
    Returns decoded string (including newline if model generates it).
    """
    # encode the input and get states
    states_value = encoder_model.predict(input_seq, verbose=0)
    # create target sequence of length 1 with the "start" token
    target_seq = np.zeros((1, 1, num_decoder_tokens), dtype="float32")
    if '\t' in target_token_index:
        target_seq[0, 0, target_token_index['\t']] = 1.0
    else:
        logger.warning("Start token '\\t' not present in target_token_index; trying index 0 instead.")
        target_seq[0, 0, 0] = 1.0

    stop_condition = False
    decoded_sentence = ""
    steps = 0
    while not stop_condition and steps < max_decoder_seq_length_given:
        # decoder_model expects: [decoder_input] + [state_h, state_c]...
        preds_and_states = decoder_model.predict([target_seq] + states_value, verbose=0)
        # preds_and_states is [decoder_outputs] + new states
        output_tokens = preds_and_states[0]
        new_states = preds_and_states[1:]
        # take highest-prob token
        sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))
        sampled_char = reverse_target_char_index.get(sampled_token_index, "")
        decoded_sentence += sampled_char
        # stop criteria
        if sampled_char == '\n':
            stop_condition = True

        # update target_seq (the input to the decoder for next step)
        target_seq = np.zeros((1, 1, num_decoder_tokens), dtype="float32")
        target_seq[0, 0, sampled_token_index] = 1.0

        # update states_value
        states_value = new_states
        steps += 1

    return decoded_sentence


def decode_sequence_beam(input_seq, encoder_model, decoder_model,
                         num_decoder_tokens, target_token_index,
                         reverse_target_char_index, max_decoder_seq_length_given=51,
                         beam_width=3):
    """
    Decode a sequence using beam search.
    Returns decoded string (without the start token and up to newline).
    """
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Beam entries: (log_prob, decoded_sequence_str, states)
    beam = [(0.0, '\t', states_value)] # starting w/ start token
    completed = []

    for step in range(max_decoder_seq_length_given):
        new_beam = []
        for log_prob, seq, states in beam:
            # obtain one-hot for last char in seq
            last_char = seq[-1]
            if last_char not in target_token_index:
                # safety: if last_char is not in index, we skip this hypothesis
                continue
            target_seq = np.zeros((1, 1, num_decoder_tokens), dtype="float32")
            target_seq[0, 0, target_token_index[last_char]] = 1.0
            preds = decoder_model.predict([target_seq] + states, verbose=0)
            output_tokens = preds[0]
            # new states may vary by model, typical: (h, c)
            new_states = preds[1:]

            # top K token probabilities
            top_indices = np.argsort(output_tokens[0, -1, :])[-beam_width:]
            for idx in top_indices:
                char = reverse_target_char_index.get(int(idx), "")
                prob = float(output_tokens[0, -1, idx])
                new_log_prob = log_prob + np.log(prob + 1e-12)
                new_seq = seq + char
                if char == '\n':
                    completed.append((new_log_prob, new_seq))
                else:
                    new_beam.append((new_log_prob, new_seq, new_states))

        # top beam_width sequences kept
        beam = heapq.nlargest(beam_width, new_beam, key=lambda x: x[0])
        if not beam:
            break

    # if there are completed, pick best; otherwise pick best current beam
    if completed:
        best_seq = max(completed, key=lambda x: x[0])[1]
    elif beam:
        best_seq = max(beam, key=lambda x: x[0])[1]
    else:
        best_seq = ''

    # remove start token and stop at newline
    return best_seq.strip('\t').split('\n')[0]  

 
# Batch evaluation & helper functions to test beam without retraining 
def levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[la][lb]


def cer(ref: str, hyp: str) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return levenshtein(ref, hyp) / len(ref)


def exact_match(ref: str, hyp: str) -> bool:
    return ref == hyp


def evaluate_batch(sent_pairs, encoder_model, decoder_model, input_token_index,
                   max_encoder_seq_length, num_encoder_tokens, num_decoder_tokens,
                   target_token_index, reverse_target_char_index,
                   max_decoder_len, beam_width=3, save_path=None):
    """
    sent_pairs: list of (src, ref) tuples (ref should not include the \t or \n if possible)
    Returns (metrics dict, results list)
    """
    results = []
    for src, ref in sent_pairs:
        input_seq = encoding_sentence_to_predict(src, input_token_index, max_encoder_seq_length, num_encoder_tokens)
        if beam_width and beam_width > 1:
            pred = decode_sequence_beam(input_seq, encoder_model, decoder_model,
                                        num_decoder_tokens, target_token_index, reverse_target_char_index,
                                        max_decoder_seq_length_given=max_decoder_len, beam_width=beam_width)
        else:
            pred = decode_sequence_inference(input_seq, encoder_model, decoder_model,
                                             num_decoder_tokens, target_token_index, reverse_target_char_index,
                                             max_decoder_seq_length_given=max_decoder_len)
        pred = pred.strip()
        ref_clean = ref.strip()
        results.append({"source": src, "reference": ref_clean, "prediction": pred})

    # metrics
    num = len(results)
    avg_cer = float(np.mean([cer(r["reference"], r["prediction"]) for r in results])) if num else 0.0
    exact = sum(1 for r in results if exact_match(r["reference"], r["prediction"]))
    exact_acc = exact / num if num else 0.0
    avg_pred_len = float(np.mean([len(r["prediction"]) for r in results])) if num else 0.0
    avg_ref_len = float(np.mean([len(r["reference"]) for r in results])) if num else 0.0

    metrics = {
        "num_samples": num,
        "average_CER": avg_cer,
        "exact_match_accuracy": exact_acc,
        "avg_pred_length": avg_pred_len,
        "avg_ref_length": avg_ref_len,
        "beam_width": beam_width,
        "timestamp": datetime.datetime.now().isoformat()
    }

    logger.info("Batch evaluation results: %s", json.dumps(metrics, indent=2))
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        preds_file = os.path.join(save_path, "batch_predictions.jsonl")
        metrics_file = os.path.join(save_path, "batch_metrics.json")
        with open(preds_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(metrics_file, "w", encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("Saved batch predictions to %s and metrics to %s", preds_file, metrics_file)

    return metrics, results


def try_auto_build_inference(trained_model_path, encoder_out_path, decoder_out_path):
    """
    Try to load the full trained model and build encoder & decoder inference models.
    This mirrors the generateInferenceModel logic used in training code.
    Returns (encoder_model, decoder_model) or (None, None) if fails.
    """
    if not os.path.exists(trained_model_path):
        logger.warning("No full trained model found at %s, cannot auto-build inference models.", trained_model_path)
        return None, None
    try:
        logger.info("Attempting to auto-build inference models from %s", trained_model_path)
        trained = keras.models.load_model(trained_model_path)
        # The following indexing assumes the standard layer order used in the training script:
        encoder_inputs = trained.input[0]
        encoder_outputs, state_h_enc, state_c_enc = trained.layers[2].output
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model_local = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = trained.input[1]
        decoder_state_input_h = keras.Input(shape=(state_h_enc.shape[-1],))
        decoder_state_input_c = keras.Input(shape=(state_c_enc.shape[-1],))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = trained.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = trained.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model_local = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        # save built inference models
        encoder_model_local.save(encoder_out_path, overwrite=True)
        decoder_model_local.save(decoder_out_path, overwrite=True)
        logger.info("Auto-built and saved encoder to %s and decoder to %s", encoder_out_path, decoder_out_path)
        return encoder_model_local, decoder_model_local
    except Exception as e:
        logger.exception("Auto-building inference models failed: %s", str(e))
        return None, None


# -------------------------
# CLI:
def parse_args():
    p = argparse.ArgumentParser(description="Char-level seq2seq inference CLI (en-de / de-en).")
    p.add_argument("-s", "--input_sentence", type=str,
                   help="Input sentence to translate. If absent and not in interactive mode, program exits.")
    p.add_argument("--direction", choices=["en-de", "de-en"], default="en-de",
                   help="Translation direction. You must provide matching models/encodings for the direction.")
    p.add_argument("--char2enc", type=str, default=DEFAULT_CHAR2ENC,
                   help="Path to char2encoding pickle that matches the chosen models.")
    p.add_argument("--encoder", type=str, default=DEFAULT_ENCODER,
                   help="Path to saved encoder inference model (keras).")
    p.add_argument("--decoder", type=str, default=DEFAULT_DECODER,
                   help="Path to saved decoder inference model (keras).")
    p.add_argument("--max_dec_len", type=int, default=None,
                   help="Max decoder length (optional). If not provided, defaults to 2*max_encoder_seq_length or 100, whichever bigger.")
    p.add_argument("--interactive", action="store_true", help="Interactive mode (prompt repeatedly).")
    p.add_argument("--quiet", action="store_true", help="Suppress informational logs.")

    # new flags
    p.add_argument("--sample-eval", action="store_true", help="Run beam decode on built-in sample sentences to test immediately.")
    p.add_argument("--test-file", type=str, default=None, help="Path to TSV file with 'src<TAB>ref' lines for batch evaluation.")
    p.add_argument("--save-eval-dir", type=str, default=None, help="Directory to save batch predictions/metrics if doing test-file or sample-eval.")
    p.add_argument("--auto-build-inference", action="store_true",
                   help="If encoder/decoder models missing, try to build them from sq2sq_model.keras")
    p.add_argument("--beam-width", type=int, default=3, help="Beam search width; use 1 for greedy decoding.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quiet:
        logger.setLevel(logging.WARNING)

    # char2encoding check (required)
    if not os.path.exists(args.char2enc):
        logger.error("char2encoding file not found: %s", args.char2enc)
        sys.exit(2)

    logger.info("Loading char2encoding from %s", args.char2enc)
    try:
        input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index = load_char2encoding(args.char2enc)
    except Exception as e:
        logger.exception("Failed to load char2encoding: %s", str(e))
        sys.exit(2)

    # If requested, try to auto-build inference models BEFORE complaining about missing files
    sq2sq_model_path = os.path.join(os.path.dirname(args.encoder), "sq2sq_model.keras")
    if args.auto_build_inference and (not os.path.exists(args.encoder) or not os.path.exists(args.decoder)):
        built_enc, built_dec = try_auto_build_inference(sq2sq_model_path, args.encoder, args.decoder)
        if built_enc is None:
            logger.warning("Auto-build requested but failed; please run the training pipeline's generateInferenceModel or provide encoder/decoder model files.")
        else:
            encoder_model = built_enc
            decoder_model = built_dec

    # Load encoder/decoder models (if not built above)
    # Encoder
    if 'encoder_model' not in locals():
        if not os.path.exists(args.encoder):
            logger.error("Encoder model not found: %s (use --auto-build-inference to attempt building from sq2sq_model.keras)", args.encoder)
            sys.exit(2)
        logger.info("Loading encoder model from %s", args.encoder)
        encoder_model = keras.models.load_model(args.encoder)

    # Decoder
    if 'decoder_model' not in locals():
        if not os.path.exists(args.decoder):
            logger.error("Decoder model not found: %s (use --auto-build-inference to attempt building from sq2sq_model.keras)", args.decoder)
            sys.exit(2)
        logger.info("Loading decoder model from %s", args.decoder)
        decoder_model = keras.models.load_model(args.decoder)

    # derive sensible max_dec_len if not provided
    if args.max_dec_len is None:
        max_dec = max(100, max_encoder_seq_length * 2)
    else:
        max_dec = args.max_dec_len

    # translator helper (uses beam if beam_width > 1)
    def translate_one(sentence):
        input_seq = encoding_sentence_to_predict(sentence, input_token_index, max_encoder_seq_length, num_encoder_tokens)
        if args.beam_width and args.beam_width > 1:
            decoded = decode_sequence_beam(input_seq, encoder_model, decoder_model,
                                           num_decoder_tokens, target_token_index, reverse_target_char_index,
                                           max_decoder_seq_length_given=max_dec, beam_width=args.beam_width)
        else:
            decoded = decode_sequence_inference(input_seq, encoder_model, decoder_model,
                                                num_decoder_tokens, target_token_index, reverse_target_char_index,
                                                max_decoder_seq_length_given=max_dec)
        return decoded.strip()

    # sample-eval
    if args.sample_eval:
        # sample_pairs = [
        #     ("I loved her.", "Ich liebte sie."),
        #     ("I can't be late.", "Ich darf nicht zu spät sein."),
        #     ("It may snow today.", "Es könnte heute schneien."),
        #     ("He studied abroad.", "Er hat im Ausland studiert."),
        #     ("My bags are packed.", "Meine Taschen sind gepackt."),
        #     ("Sorry for your loss.", "Mein herzliches Beileid."),
        # ]
        sample_pairs = [
            ("Kannst du mir das Salz reichen?", "Could you pass me the salt?"),
            ("Kommst du heute Abend zur Party?", "Are you coming to the party tonight?"),
            ("Hast du etwas dagegen, wenn ich das Fenster öffne?", "Do you mind if I open the window?"),
            ("Ich freue mich auf das Wochenende.", "I'm looking forward to the weekend."),
            ("Sie lernt seit zwei Jahren Deutsch.", "She has been studying German for two years."),
            ("Wir sollten einen Arzt rufen.", "We should call a doctor."),
            ("Er konnte seine Schlüssel nirgendwo finden.", "He couldn't find his keys anywhere."),
            ("Es fängt an zu regnen.", "It's starting to rain."),
            ("Sie werden morgen früh ankommen.", "They will arrive tomorrow morning."),
            ("Ich habe diesen Film noch nie gesehen.", "I have never seen that movie.")
        ]
        logger.info("Running sample-eval with beam_width=%s", args.beam_width)
        save_dir = args.save_eval_dir if args.save_eval_dir else None
        metrics, res = evaluate_batch(sample_pairs, encoder_model, decoder_model, input_token_index,
                                      max_encoder_seq_length, num_encoder_tokens, num_decoder_tokens,
                                      target_token_index, reverse_target_char_index, max_dec,
                                      beam_width=args.beam_width, save_path=save_dir)
        for r in res[:10]:
            print(json.dumps(r, ensure_ascii=False))
        return

    # test-file evaluation
    if args.test_file:
        if not os.path.exists(args.test_file):
            logger.error("test-file not found: %s", args.test_file)
            sys.exit(2)
        pairs = []
        with open(args.test_file, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                if "\t" in ln:
                    src, ref = ln.split("\t", 1)
                else:
                    src = ln
                    ref = ""
                pairs.append((src, ref))
        save_dir = args.save_eval_dir if args.save_eval_dir else None
        metrics, res = evaluate_batch(pairs, encoder_model, decoder_model, input_token_index,
                                      max_encoder_seq_length, num_encoder_tokens, num_decoder_tokens,
                                      target_token_index, reverse_target_char_index, max_dec,
                                      beam_width=args.beam_width, save_path=save_dir)
        print(json.dumps(metrics, indent=2))
        return

    # Interactive or single-shot
    if args.interactive:
        logger.info("Interactive mode. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                sent = args.input_sentence if args.input_sentence else input("Input> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not sent:
                continue
            if sent.lower() in {"quit", "exit"}:
                break
            out = translate_one(sent)
            print("\nInput :", sent)
            print("Output:", out)
            args.input_sentence = None
        logger.info("Exiting interactive mode.")
        return

    # Non-interactive: must provide -s
    if not args.input_sentence:
        logger.error("No input sentence provided. Use -s / --input_sentence or --interactive.")
        sys.exit(2)

    out = translate_one(args.input_sentence)
    print("\n------------------------------------------\n")
    print("Input sentence:", args.input_sentence)
    print("Decoded sentence:", out)
    print("\n------------------------------------------\n")


if __name__ == "__main__":
    main()

 
# CLI Examples 
# 1) Quick sample test (no file needed):
# python infer_translate.py --sample-eval --char2enc r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\char2encoding.pkl" \
#   --encoder r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\encoder_model.keras" \
#   --decoder r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\decoder_model.keras" \
#   --beam-width 3
# py -m src.infer_translate --sample-eval --beam-width 3

#
# 2) Evaluate a TSV test file (one src[TAB]ref per line):
# python infer_translate.py --test-file data/test_small.tsv --char2enc r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\char2encoding.pkl" \
#   --encoder r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\encoder_model.keras" \
#   --decoder r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\decoder_model.keras" \
#   --save-eval-dir experiments/quick_eval_beam3 --beam-width 3
#
# 3) If you only have sq2sq_model.keras (full model) but not encoder/decoder saved:
# python infer_translate.py --auto-build-inference --char2enc r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\char2encoding.pkl" \
#   --encoder r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\encoder_model.keras" \
#   --decoder r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\decoder_model.keras"
# then run --sample-eval or --test-file as above
#
# 4) Single-shot translation:
# python infer_translate.py -s "I loved her." --char2enc r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\char2encoding.pkl" \
#   --encoder r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\encoder_model.keras" \
#   --decoder r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\decoder_model.keras" \
#   --beam-width 3
#
# 5) Interactive mode:
# python infer_translate.py --interactive --char2enc r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\char2encoding.pkl" \
#   --encoder r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\encoder_model.keras" \
#   --decoder r"C:\Arbeit\Phoenix\LT-seq2seq\src\trained_models\decoder_model.keras" \
#   --beam-width 3
#
# 6) Use greedy decoding:
# python infer_translate.py -s "I loved her." --beam-width 1
#
# -------------------------