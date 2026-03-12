import streamlit as st
import numpy as np
import os
import pickle
import logging
from pathlib import Path
import heapq, shutil
 
try: # tensorflow.keras from tensorflow if available, fallback to keras
    from tensorflow import keras
except Exception:
    import keras 
try:
    import sentencepiece as spm
    HAVE_SPM = True
except Exception:
    HAVE_SPM = False
try: # HF hub for larger models
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("streamlit_app_infer")    
MODEL_DIR = Path(__file__).parent.parent / "trained_models" # where .keras and token_tool_*.pkl are

# fetch the req. model files from HFHub into MODEL_DIR
@st.cache_resource
def ensure_models_present(direction: str):
    """
    ensures encoder_model_{direction}.keras, decoder_model_{direction}.keras and token_tool_{direction}.pkl
    are present under src/trained_models/. If any are missing, attempt to download from a Hugging Face repo.
    Repo. selection priority:
     1) HF_REPO_{DIRECTION} (e.g. HF_REPO_DEU2ENG)
     2) HF_REPO
     3) If neither set: do nothing (function will later cause FileNotFoundError in loader)
    """
    required_files = [
        f"encoder_model_{direction}.keras",
        f"decoder_model_{direction}.keras",
        f"token_tool_{direction}.pkl",
    ]

    # quick local check
    missing = [f for f in required_files if not (MODEL_DIR / f).exists()]
    if not missing:
        logger.debug("All model files present locally.")
        return {"status": "ok", "missing": []}

    # pick repo id from env
    env_repo_key = f"HF_REPO_{direction.upper()}"
    repo_id = os.getenv(env_repo_key) or os.getenv("HF_REPO")
    hf_token = os.getenv("HF_TOKEN")  # (opt. but def. for private repos)
    if not repo_id:
        logger.debug(f"Missing files {missing} and no HF_REPO env var set.")
        return {"status": "missing_repo_env", "missing": missing}

    if snapshot_download is None:
        logger.error("huggingface_hub.snapshot_download not available; please install huggingface-hub")
        return {"status": "hf_not_installed", "missing": missing}

    st.info(f"Missing files: {', '.join(missing)} — attempting download from Hugging Face repo `{repo_id}`")
    try:
        # download the entire repo snapshot (huggingface_hub caches)
        dl_dir = snapshot_download(repo_id, use_auth_token=hf_token)
    except Exception as e:
        logger.exception("snapshot_download failed")
        return {"status": "download_failed", "error": str(e), "missing": missing}

    dl_path = Path(dl_dir)
    copied = []
    not_found = []
    # attempt to locate each required file in downloaded repo (search recursively)
    for fname in required_files:
        # first check root
        candidate = dl_path / fname
        if candidate.exists():
            dest = MODEL_DIR / fname
            shutil.copy2(candidate, dest)
            copied.append(fname)
            continue
        # else search anywhere inside the downloaded snapshot
        found = None
        for p in dl_path.rglob(fname):
            found = p
            break
        if found:
            shutil.copy2(found, MODEL_DIR / fname)
            copied.append(fname)
        else:
            not_found.append(fname)

    if not_found:
        logger.warning(f"Downloaded repo `{repo_id}` but could not find: {not_found}")
        return {"status": "incomplete", "missing": not_found, "copied": copied}

    logger.debug(f"Copied files from HF repo: {copied}")
    return {"status": "ok", "copied": copied}

# helpers to load resources (cached)
@st.cache_resource
def load_token_tool(direction: str):
    """
    loads token_tool_{direction}.pkl which is expected to contain (token_tool, max_enc_len, max_dec_len, input_vocab_size, target_vocab_size)
    token_tool is either:
      - ("sentencepiece", sp_model_path)            saved by train.py (all versions)
      - ("keras_tokenizer_json", json_str)          saved by updated train.py (current)
      - ("keras_tokenizer", tokenizer_obj)          legacy format from older train.py

    Downstream functions in this app will always receive one of two normalized tuples:
      - ("sentencepiece", model_path, sp_instance)  for SentencePiece
      - ("keras_tokenizer", tok)                    for Keras tokenizer (both new and legacy)
    """
    p = MODEL_DIR / f"token_tool_{direction}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"token_tool file not found: {p}")
    with open(p, "rb") as f: # loading sequentially in same order as was saved during training 
        token_tool = pickle.load(f)
        max_enc_len = pickle.load(f)
        max_dec_len = pickle.load(f)
        input_vocab_size = pickle.load(f)
        target_vocab_size = pickle.load(f)
    # if token_tool using sentencepiece: instantiate processor
    if isinstance(token_tool, tuple) and token_tool[0] == "sentencepiece":
        model_path = token_tool[1]
        if not Path(model_path).exists(): 
            alt = MODEL_DIR / Path(model_path).name
            if alt.exists():
                model_path = str(alt)
            else:
                # As a convenience: if HF snapshot placed the spm elsewhere (e.g. subfolder),
                # try to find it inside MODEL_DIR and repo cache
                # (the HF snapshot step copies only files with matching names to MODEL_DIR,
                # so the alt check should usually succeed)
                raise FileNotFoundError(f"SentencePiece model not found at {model_path} or {alt}")
        # if not HAVE_SPM:
        #     raise RuntimeError("sentencepiece not installed but token_tool requires it. `pip install sentencepiece`")
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        return ("sentencepiece", model_path, sp), max_enc_len, max_dec_len, input_vocab_size, target_vocab_size


    # since the updated trainer serializes the Keras tokenizer as a JSON string ("keras_tokenizer_json")
    # rather than pickling the raw object ("keras_tokenizer"), because raw Tokenizer objects
    # are not reliably picklable across Keras versions and the class is deprecated in newer Keras,
    # we'll deserialize back to a Tokenizer object here so all downstream functions are unchanged
    if isinstance(token_tool, tuple) and token_tool[0] == "keras_tokenizer_json":
        json_str = token_tool[1]
        try:
            tok = keras.preprocessing.text.tokenizer_from_json(json_str)
        except AttributeError:
            # tokenizer_from_json not available in very old Keras builds: fall back to eval-based restore
            logger.warning("tokenizer_from_json not available, attempting manual JSON restore")
            import json as _json
            cfg = _json.loads(json_str)
            tok = keras.preprocessing.text.Tokenizer()
            tok.word_index = cfg.get("word_index", {})
            tok.index_word = {int(v): k for k, v in tok.word_index.items()}
        return ("keras_tokenizer", tok), max_enc_len, max_dec_len, input_vocab_size, target_vocab_size

    # legacy format: raw Keras Tokenizer object was pickled directly
    if isinstance(token_tool, tuple) and token_tool[0] == "keras_tokenizer":
        tok = token_tool[1]
        # ensure tokenizer has index_word mapping (Keras Tokenizer has index_word attribute)
        return ("keras_tokenizer", tok), max_enc_len, max_dec_len, input_vocab_size, target_vocab_size

    raise RuntimeError("Unrecognized token_tool structure in token_tool pickle.")

@st.cache_resource
def load_inference_models(direction: str):
    """
    loads encoder_model_{direction}.keras and decoder_model_{direction}.keras (i.e. the inference models)
    """
    enc_path = MODEL_DIR / f"encoder_model_{direction}.keras"
    dec_path = MODEL_DIR / f"decoder_model_{direction}.keras"
    if not enc_path.exists() or not dec_path.exists():
        raise FileNotFoundError(f"Missing inference models. Expected {enc_path} and {dec_path}")
    # load w/o compiling, for speed
    encoder = keras.models.load_model(str(enc_path), compile=False)
    decoder = keras.models.load_model(str(dec_path), compile=False)
    return encoder, decoder

# enc-dec utlil.(s)
def encode_sentence_to_ids(sentence: str, token_tool_tuple, max_enc_len: int):
    """
    encode a raw string into padded token id sequence (1D numpy array) for model input,
    for SentencePiece we rely on pad_id=0, for Keras tokenizer we use 0 as pad token too
    """
    tool_type = token_tool_tuple[0]
    if tool_type == "sentencepiece":
        sp = token_tool_tuple[2] # token_tool_tuple = ("sentencepiece", model_path, sp_instance)
        ids = sp.EncodeAsIds(sentence) # sp: instance of SentencePieceProcessor
    else:
        tok = token_tool_tuple[1] # ("keras_tokenizer", tok)
        seqs = tok.texts_to_sequences([sentence]) # tok: instance of keras.preprocessing.text.Tokenizer
        ids = seqs[0] if seqs else []

    # truncate OR pad post
    if len(ids) >= max_enc_len:
        ids_trunc = ids[:max_enc_len]
    else:
        ids_trunc = ids + [0] * (max_enc_len - len(ids))
    return np.array([ids_trunc], dtype=np.int32)

def ids_to_text(decoded_ids: list, token_tool_tuple):
    """
    convert decoded id list back to text string,
        For SentencePiece: use sp.DecodeIds
        For Keras tokenizer: reconstruct text using index_word map
    """
    if token_tool_tuple[0] == "sentencepiece":
        sp = token_tool_tuple[2]
        return sp.DecodeIds(decoded_ids)
    else:
        tok = token_tool_tuple[1]
        inv = getattr(tok, "index_word", None)
        if inv is None:
            # fallback reconstruct
            inv = {v: k for k, v in tok.word_index.items()}
        words = [inv.get(i, "") for i in decoded_ids if i != 0]
        return " ".join(w for w in words if w)

def greedy_decode_token(encoder_model, decoder_model, token_tool_tuple, input_seq_ids, max_dec_len: int):
    """
    Greedy decode using inference encoder/decoder models,
    - encoder_model returns [encoder_outputs, state_h, state_c]
    - decoder_model expects [decoder_token_input, prev_h, prev_c, encoder_outputs] -> [probs, next_h, next_c]
    """
    # encode
    enc_outs, h, c = encoder_model.predict(input_seq_ids, verbose=0)
    # BOS/EOS ids
    if token_tool_tuple[0] == "sentencepiece":
        BOS_ID = 2
        EOS_ID = 3
    else:
        tok = token_tool_tuple[1]
        # try common keys
        bos_id = tok.word_index.get("<bos>") or tok.word_index.get("bos") or 1
        eos_id = tok.word_index.get("<eos>") or tok.word_index.get("eos") or 1
        BOS_ID, EOS_ID = int(bos_id), int(eos_id)
    decoded_ids = []
    cur_tok = np.array([[BOS_ID]], dtype=np.int32)
    prev_h, prev_c = h, c
    for _ in range(max_dec_len):
        preds, next_h, next_c = decoder_model.predict([cur_tok, prev_h, prev_c, enc_outs], verbose=0)
        # preds shape (1, 1, vocab)
        prob = preds[0, -1, :]
        idx = int(np.argmax(prob))
        if idx == EOS_ID:
            break
        decoded_ids.append(idx)
        cur_tok = np.array([[idx]], dtype=np.int32)
        prev_h, prev_c = next_h, next_c
    # convert ids -> text
    return ids_to_text(decoded_ids, token_tool_tuple), decoded_ids

def beam_decode_token(encoder_model, decoder_model, token_tool_tuple, input_seq_ids, max_dec_len: int, beam_width: int = 3):
    """
    beam search implementation for the token-level inference decoder,
        Beam entries: (log_prob, token_list, prev_h, prev_c)
    We keep states for each beam (they come from decoder predictions)
    """
    enc_outs, h0, c0 = encoder_model.predict(input_seq_ids, verbose=0)
    # BOS/EOS ids
    if token_tool_tuple[0] == "sentencepiece":
        BOS_ID, EOS_ID = 2, 3
    else:
        tok = token_tool_tuple[1]
        bos_id = tok.word_index.get("<bos>") or tok.word_index.get("bos") or 1
        eos_id = tok.word_index.get("<eos>") or tok.word_index.get("eos") or 1
        BOS_ID, EOS_ID = int(bos_id), int(eos_id)

    # initial beam
    beam = [(0.0, [BOS_ID], h0, c0)]
    completed = []

    for _ in range(max_dec_len):
        candidates = []

        # Separate predict call per beam (improved on w/ single batch below)
        # for logp, seq_ids, prev_h, prev_c in beam:
        #     last = seq_ids[-1]
        #     if last == EOS_ID:
        #         # already completed: keep as is
        #         completed.append((logp, seq_ids))
        #         continue
        #     cur_tok = np.array([[last]], dtype=np.int32)
        #     preds, next_h, next_c = decoder_model.predict([cur_tok, prev_h, prev_c, enc_outs], verbose=0)
        #     probs = preds[0, -1, :] # (vocab,)
        #     # take top-K
        #     topk = np.argsort(probs)[-beam_width:]
        #     for idx in topk:
        #         p = float(probs[idx])
        #         new_logp = logp + np.log(p + 1e-12)
        #         new_seq = seq_ids + [int(idx)]
        #         candidates.append((new_logp, new_seq, next_h, next_c))

        # Single batched predict for all active beams
        # split completed beams out before batching
        active = [(lp, seq, h, c) for lp, seq, h, c in beam if seq[-1] != EOS_ID]
        completed.extend([(lp, seq) for lp, seq, h, c in beam if seq[-1] == EOS_ID])
        if not active: break
        n_active = len(active)
        # tile enc_outs once for all active beams -> (n_active, seq_len, hidden)
        enc_outs_tiled = np.tile(enc_outs, (n_active, 1, 1))
        # batch all active beam inputs batched
        cur_toks = np.array([[seq[-1]] for _, seq, _, _ in active], dtype=np.int32) # (n_active, 1)
        hs = np.concatenate([h for _, _, h, _ in active], axis=0) # (n_active, hidden)
        cs = np.concatenate([c for _, _, _, c in active], axis=0) # (n_active, hidden)
        # single forward pass for ALL beams
        preds, next_hs, next_cs = decoder_model.predict(
            [cur_toks, hs, cs, enc_outs_tiled], verbose=0
        ) # preds: (n_active, 1, vocab)
        for i, (logp, seq_ids, _, _) in enumerate(active):
            probs = preds[i, -1, :]
            topk = np.argsort(probs)[-beam_width:]
            next_h_i = next_hs[i : i + 1] # keep dims: (1, hidden)
            next_c_i = next_cs[i : i + 1]
            for idx in topk:
                p = float(probs[idx])
                new_logp = logp + np.log(p + 1e-12)
                candidates.append((new_logp, seq_ids + [int(idx)], next_h_i, next_c_i))
        # top beam_width candidates choosen
        if not candidates:
            break
        beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])

        # # stop early if many completed
        # if len(completed) >= beam_width:
        #     break
        # stop only when best completed sequence beats all active beams
        # (no active beam can ever outscore it, so further expansion is pointless)
        if completed:
            best_completed_score = max(c[0] for c in completed)
            best_active_score = max(b[0] for b in beam)
            if best_completed_score >= best_active_score:
                break


    # choose best completed if any, else best partial
    if completed:
        best = max(completed, key=lambda x: x[0])[1]
    else:
        best = max(beam, key=lambda x: x[0])[1]

    # remove leading BOS and trailing EOS if present
    res_ids = [i for i in best if i not in (BOS_ID, EOS_ID)]
    return ids_to_text(res_ids, token_tool_tuple), res_ids

# ui
st.set_page_config(page_title="Seq2Seq MT - a Phoenix model", layout="centered")
st.title("Machine Translation (Seq2Seq)")
st.markdown(
    """
    Runs inference using token-level encoder/decoder + attention models,
    \nmodels expected in `trained_models/`:-
    \n- encoder_model_{direction}.keras
    \n- decoder_model_{direction}.keras
    \n- token_tool_{direction}.pkl (contains token_tool metadata)
    \nDirections being: `deu2eng` OR `eng2deu`
    """
)

direction_human = st.radio("Translation direction", ("Deutsch -> Englisch (deu2eng)", "English -> German (eng2deu)"))
direction = "deu2eng" if "deu2eng" in direction_human else "eng2deu"

# ensure model files exist (local or from HF)
hf_result = ensure_models_present(direction)
if hf_result.get("status") == "missing_repo_env":
    st.info(
        "Some model files are missing locally. "
        "If you want Streamlit to download them automatically, set environment variable HF_REPO or HF_REPO_{DIRECTION} "
        "(e.g. HF_REPO_DEU2ENG) to a Hugging Face repo id that contains the files. "
        "Set HF_TOKEN for private repos."
    )
elif hf_result.get("status") == "hf_not_installed":
    st.error("The huggingface-hub package is not installed in this environment. Run `pip install huggingface-hub`.")
elif hf_result.get("status") == "download_failed":
    st.error(f"Hugging Face download failed: {hf_result.get('error')}")
elif hf_result.get("status") == "incomplete":
    st.error(f"Downloaded repo but missing files: {hf_result.get('missing')}. Copied: {hf_result.get('copied', [])}")

# loading resources
try:
    token_tool_tuple, max_enc_len, max_dec_len, input_vocab_size, target_vocab_size = load_token_tool(direction)
    encoder_model, decoder_model = load_inference_models(direction)
except Exception as e:
    st.error(f"Failed to load models or token tool: {e}")
    logger.exception("Model load error")
    raise

st.sidebar.markdown("### Inference options")
decode_method = st.sidebar.selectbox("Decoding technique to use..", ("greedy", "beam"))
beam_width = st.sidebar.slider("Beam width (when using beam)", min_value=2, max_value=12, value=3, step=1)
show_ids = st.sidebar.checkbox("Show decoded token ids", value=False)
max_decode_len = st.sidebar.number_input("Max decode length", min_value=10, max_value=500, value=int(max_dec_len))

# input box
text = st.text_area("Enter text to translate", value="", height=120)

if st.button("Translate"):
    if not text.strip():
        st.warning("Don't you forget to put in the sentence to translate dummy ;)")
    else:
        try:
            input_seq = encode_sentence_to_ids(text, token_tool_tuple, max_enc_len)
            if decode_method == "greedy":
                out_text, out_ids = greedy_decode_token(encoder_model, decoder_model, token_tool_tuple, input_seq, max_decode_len)
            else:
                out_text, out_ids = beam_decode_token(encoder_model, decoder_model, token_tool_tuple, input_seq, max_decode_len, beam_width=beam_width)
            st.subheader("Translation:-")
            st.success(out_text)
            if show_ids:
                st.write({"token_ids": out_ids})
        except Exception as e:
            st.error(f"Inference failed: {e}")
            logger.exception("Inference error")

st.markdown("---")
st.markdown("Model files found & available:-")
st.write(str(MODEL_DIR / f"encoder_model_{direction}.keras"))
st.write(str(MODEL_DIR / f"decoder_model_{direction}.keras"))
st.write(str(MODEL_DIR / f"token_tool_{direction}.pkl"))