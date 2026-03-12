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
    from huggingface_hub import snapshot_download, hf_hub_download
except Exception:
    snapshot_download = None
    hf_hub_download = None

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("streamlit_app_infer")    
MODEL_DIR = Path(__file__).parent.parent / "trained_models" # where .keras and token_tool_*.pkl are

# SPM model filename pattern: spm_{direction}.model
def _spm_filename(direction: str) -> str:
    return f"spm_{direction}.model"

# fetch the req. model files from HFHub into MODEL_DIR
@st.cache_resource
def ensure_models_present(direction: str):
    """
    Ensures all required model files (including the SentencePiece .model file) are present
    under src/trained_models/. If any are missing, attempts download from a Hugging Face repo.

    Required files:
      - encoder_model_{direction}.keras
      - decoder_model_{direction}.keras
      - token_tool_{direction}.pkl
      - spm_{direction}.model          <-- SentencePiece model (was missing from old version)

    Repo selection priority:
      1) HF_REPO_{DIRECTION} (e.g. HF_REPO_DEU2ENG)
      2) HF_REPO
      3) If neither set: do nothing (FileNotFoundError will surface later)
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        f"encoder_model_{direction}.keras",
        f"decoder_model_{direction}.keras",
        f"token_tool_{direction}.pkl",
        _spm_filename(direction), 
    ]

    # quick local check
    missing = [f for f in required_files if not (MODEL_DIR / f).exists()]
    if not missing:
        logger.debug("All model files present locally.")
        return {"status": "ok", "missing": []}

    # pick repo id from env
    env_repo_key = f"HF_REPO_{direction.upper()}"
    repo_id = os.getenv(env_repo_key) or os.getenv("HF_REPO")
    hf_token = os.getenv("HF_TOKEN")  # optional but required for private repos
    if not repo_id:
        logger.debug(f"Missing files {missing} and no HF_REPO env var set.")
        return {"status": "missing_repo_env", "missing": missing}

    if snapshot_download is None:
        logger.error("huggingface_hub not available; please install huggingface-hub")
        return {"status": "hf_not_installed", "missing": missing}

    st.info(f"Missing files: {', '.join(missing)} — downloading from Hugging Face repo `{repo_id}`")
    try:
        # snapshot_download handles Git LFS files automatically
        dl_dir = snapshot_download(
            repo_id,
            token=hf_token,
            ignore_patterns=["*.git", "*.gitattributes"],  # skip repo metadata
        )
    except Exception as e:
        logger.exception("snapshot_download failed")
        return {"status": "download_failed", "error": str(e), "missing": missing}

    dl_path = Path(dl_dir)
    copied = []
    not_found = []

    for fname in missing:  # only copy what's actually missing
        # check repo root first
        candidate = dl_path / fname
        if candidate.exists():
            shutil.copy2(candidate, MODEL_DIR / fname)
            copied.append(fname)
            continue
        # recursive search inside snapshot (handles repos that store files in subdirs)
        found = next(dl_path.rglob(fname), None)
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
    Loads token_tool_{direction}.pkl.
    Handles three token_tool variants:
      - ("sentencepiece", sp_model_path)       — all train.py versions
      - ("keras_tokenizer_json", json_str)      — updated train.py
      - ("keras_tokenizer", tokenizer_obj)      — legacy train.py

    FIX: the stored spm model_path may be a Windows absolute path from the training
    machine (e.g. C:\\Arbeit\\...). We now resolve it purely by filename, looking only
    inside MODEL_DIR (and the standard alt location), ignoring the stored absolute path.
    """
    p = MODEL_DIR / f"token_tool_{direction}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"token_tool file not found: {p}")
    with open(p, "rb") as f:
        token_tool        = pickle.load(f)
        max_enc_len       = pickle.load(f)
        max_dec_len       = pickle.load(f)
        input_vocab_size  = pickle.load(f)
        target_vocab_size = pickle.load(f)

    # sp path
    if isinstance(token_tool, tuple) and token_tool[0] == "sentencepiece":
        stored_path = token_tool[1]

        # resolve path by filename only — the stored path may be a Windows
        # absolute path (C:\...) which is meaningless on the deployment server.
        # Priority:
        #   1. Canonical name  spm_{direction}.model  in MODEL_DIR
        #   2. Basename of whatever path was stored   in MODEL_DIR
        #   3. Stored path as-is (only if it happens to exist — local dev)
        spm_by_direction = MODEL_DIR / _spm_filename(direction)
        spm_by_basename  = MODEL_DIR / Path(stored_path).name

        if spm_by_direction.exists():
            resolved_path = str(spm_by_direction)
        elif spm_by_basename.exists():
            resolved_path = str(spm_by_basename)
        elif Path(stored_path).exists():
            resolved_path = stored_path  # local dev: path is valid as-is
        else:
            raise FileNotFoundError(
                f"SentencePiece model not found.\n"
                f"  Tried (canonical): {spm_by_direction}\n"
                f"  Tried (basename):  {spm_by_basename}\n"
                f"  Stored path was:   {stored_path}\n"
                f"Make sure '{_spm_filename(direction)}' is in your HF repo "
                f"or set HF_REPO / HF_REPO_{direction.upper()} so it can be downloaded."
            )

        sp = spm.SentencePieceProcessor()
        sp.Load(resolved_path)
        return ("sentencepiece", resolved_path, sp), max_enc_len, max_dec_len, input_vocab_size, target_vocab_size

    # keras tokenizer (JSON-serialised, current format) 
    if isinstance(token_tool, tuple) and token_tool[0] == "keras_tokenizer_json":
        json_str = token_tool[1]
        try:
            tok = keras.preprocessing.text.tokenizer_from_json(json_str)
        except AttributeError:
            logger.warning("tokenizer_from_json not available, attempting manual JSON restore")
            import json as _json
            cfg = _json.loads(json_str)
            tok = keras.preprocessing.text.Tokenizer()
            tok.word_index = cfg.get("word_index", {})
            tok.index_word = {int(v): k for k, v in tok.word_index.items()}
        return ("keras_tokenizer", tok), max_enc_len, max_dec_len, input_vocab_size, target_vocab_size

    # keras tokenizer (legacy pickled object)  
    if isinstance(token_tool, tuple) and token_tool[0] == "keras_tokenizer":
        tok = token_tool[1]
        return ("keras_tokenizer", tok), max_enc_len, max_dec_len, input_vocab_size, target_vocab_size

    raise RuntimeError("Unrecognized token_tool structure in token_tool pickle.")

@st.cache_resource
def load_inference_models(direction: str):
    """
    Loads encoder_model_{direction}.keras and decoder_model_{direction}.keras.
    """
    enc_path = MODEL_DIR / f"encoder_model_{direction}.keras"
    dec_path = MODEL_DIR / f"decoder_model_{direction}.keras"
    if not enc_path.exists() or not dec_path.exists():
        raise FileNotFoundError(f"Missing inference models. Expected:\n  {enc_path}\n  {dec_path}")
    encoder = keras.models.load_model(str(enc_path), compile=False)
    decoder = keras.models.load_model(str(dec_path), compile=False)
    return encoder, decoder

# enc-dec utils
def encode_sentence_to_ids(sentence: str, token_tool_tuple, max_enc_len: int):
    tool_type = token_tool_tuple[0]
    if tool_type == "sentencepiece":
        sp = token_tool_tuple[2]
        ids = sp.EncodeAsIds(sentence)
    else:
        tok = token_tool_tuple[1]
        seqs = tok.texts_to_sequences([sentence])
        ids = seqs[0] if seqs else []

    if len(ids) >= max_enc_len:
        ids_trunc = ids[:max_enc_len]
    else:
        ids_trunc = ids + [0] * (max_enc_len - len(ids))
    return np.array([ids_trunc], dtype=np.int32)

def ids_to_text(decoded_ids: list, token_tool_tuple):
    if token_tool_tuple[0] == "sentencepiece":
        sp = token_tool_tuple[2]
        return sp.DecodeIds(decoded_ids)
    else:
        tok = token_tool_tuple[1]
        inv = getattr(tok, "index_word", None)
        if inv is None:
            inv = {v: k for k, v in tok.word_index.items()}
        words = [inv.get(i, "") for i in decoded_ids if i != 0]
        return " ".join(w for w in words if w)

def greedy_decode_token(encoder_model, decoder_model, token_tool_tuple, input_seq_ids, max_dec_len: int):
    enc_outs, h, c = encoder_model.predict(input_seq_ids, verbose=0)
    if token_tool_tuple[0] == "sentencepiece":
        BOS_ID, EOS_ID = 2, 3
    else:
        tok = token_tool_tuple[1]
        bos_id = tok.word_index.get("<bos>") or tok.word_index.get("bos") or 1
        eos_id = tok.word_index.get("<eos>") or tok.word_index.get("eos") or 1
        BOS_ID, EOS_ID = int(bos_id), int(eos_id)
    decoded_ids = []
    cur_tok = np.array([[BOS_ID]], dtype=np.int32)
    prev_h, prev_c = h, c
    for _ in range(max_dec_len):
        preds, next_h, next_c = decoder_model.predict([cur_tok, prev_h, prev_c, enc_outs], verbose=0)
        prob = preds[0, -1, :]
        idx = int(np.argmax(prob))
        if idx == EOS_ID:
            break
        decoded_ids.append(idx)
        cur_tok = np.array([[idx]], dtype=np.int32)
        prev_h, prev_c = next_h, next_c
    return ids_to_text(decoded_ids, token_tool_tuple), decoded_ids

def beam_decode_token(encoder_model, decoder_model, token_tool_tuple, input_seq_ids, max_dec_len: int, beam_width: int = 3):
    enc_outs, h0, c0 = encoder_model.predict(input_seq_ids, verbose=0)
    if token_tool_tuple[0] == "sentencepiece":
        BOS_ID, EOS_ID = 2, 3
    else:
        tok = token_tool_tuple[1]
        bos_id = tok.word_index.get("<bos>") or tok.word_index.get("bos") or 1
        eos_id = tok.word_index.get("<eos>") or tok.word_index.get("eos") or 1
        BOS_ID, EOS_ID = int(bos_id), int(eos_id)

    beam = [(0.0, [BOS_ID], h0, c0)]
    completed = []

    for _ in range(max_dec_len):
        candidates = []
        active    = [(lp, seq, h, c) for lp, seq, h, c in beam if seq[-1] != EOS_ID]
        completed.extend([(lp, seq) for lp, seq, h, c in beam if seq[-1] == EOS_ID])
        if not active:
            break
        n_active = len(active)
        enc_outs_tiled = np.tile(enc_outs, (n_active, 1, 1))
        cur_toks = np.array([[seq[-1]] for _, seq, _, _ in active], dtype=np.int32)
        hs = np.concatenate([h for _, _, h, _ in active], axis=0)
        cs = np.concatenate([c for _, _, _, c in active], axis=0)
        preds, next_hs, next_cs = decoder_model.predict(
            [cur_toks, hs, cs, enc_outs_tiled], verbose=0
        )
        for i, (logp, seq_ids, _, _) in enumerate(active):
            probs = preds[i, -1, :]
            topk = np.argsort(probs)[-beam_width:]
            next_h_i = next_hs[i : i + 1]
            next_c_i = next_cs[i : i + 1]
            for idx in topk:
                p = float(probs[idx])
                new_logp = logp + np.log(p + 1e-12)
                candidates.append((new_logp, seq_ids + [int(idx)], next_h_i, next_c_i))
        if not candidates:
            break
        beam = heapq.nlargest(beam_width, candidates, key=lambda x: x[0])
        if completed:
            best_completed_score = max(c[0] for c in completed)
            best_active_score    = max(b[0] for b in beam)
            if best_completed_score >= best_active_score:
                break

    if completed:
        best = max(completed, key=lambda x: x[0])[1]
    else:
        best = max(beam, key=lambda x: x[0])[1]

    res_ids = [i for i in best if i not in (BOS_ID, EOS_ID)]
    return ids_to_text(res_ids, token_tool_tuple), res_ids

# ui
st.set_page_config(page_title="Seq2Seq MT - a Phoenix model", layout="centered")
st.title("Machine Translation (Seq2Seq)")
st.markdown(
    """
    Runs inference using token-level encoder/decoder + attention models.  
    Models expected in `trained_models/`:
    - `encoder_model_{direction}.keras`
    - `decoder_model_{direction}.keras`
    - `token_tool_{direction}.pkl`
    - `spm_{direction}.model` *(SentencePiece vocab)*

    Directions: `deu2eng` or `eng2deu`
    """
)

direction_human = st.radio("Translation direction", ("Deutsch -> Englisch (deu2eng)", "English -> German (eng2deu)"))
direction = "deu2eng" if "deu2eng" in direction_human else "eng2deu"

# ensure model files exist (local or downloaded from HF Hub)
hf_result = ensure_models_present(direction)
if hf_result.get("status") == "missing_repo_env":
    st.info(
        "Some model files are missing locally. "
        "Set the environment variable **HF_REPO** (or **HF_REPO_DEU2ENG** / **HF_REPO_ENG2DEU**) "
        "to a Hugging Face repo id that contains the model files, and optionally **HF_TOKEN** for private repos."
    )
elif hf_result.get("status") == "hf_not_installed":
    st.error("The `huggingface-hub` package is not installed. Run `pip install huggingface-hub`.")
elif hf_result.get("status") == "download_failed":
    st.error(f"Hugging Face download failed: {hf_result.get('error')}")
elif hf_result.get("status") == "incomplete":
    st.error(f"Downloaded repo but still missing: {hf_result.get('missing')}. Copied: {hf_result.get('copied', [])}")

# load resources
try:
    token_tool_tuple, max_enc_len, max_dec_len, input_vocab_size, target_vocab_size = load_token_tool(direction)
    encoder_model, decoder_model = load_inference_models(direction)
except Exception as e:
    st.error(f"Failed to load models or token tool: {e}")
    logger.exception("Model load error")
    raise

st.sidebar.markdown("### Inference options")
decode_method   = st.sidebar.selectbox("Decoding technique", ("greedy", "beam"))
beam_width      = st.sidebar.slider("Beam width (beam only)", min_value=2, max_value=12, value=3, step=1)
show_ids        = st.sidebar.checkbox("Show decoded token ids", value=False)
max_decode_len  = st.sidebar.number_input("Max decode length", min_value=10, max_value=500, value=int(max_dec_len))

text = st.text_area("Enter text to translate", value="", height=120)

if st.button("Translate"):
    if not text.strip():
        st.warning("Please enter a sentence to translate.")
    else:
        try:
            input_seq = encode_sentence_to_ids(text, token_tool_tuple, max_enc_len)
            if decode_method == "greedy":
                out_text, out_ids = greedy_decode_token(
                    encoder_model, decoder_model, token_tool_tuple, input_seq, max_decode_len)
            else:
                out_text, out_ids = beam_decode_token(
                    encoder_model, decoder_model, token_tool_tuple, input_seq, max_decode_len,
                    beam_width=beam_width)
            st.subheader("Translation:")
            st.success(out_text)
            if show_ids:
                st.write({"token_ids": out_ids})
        except Exception as e:
            st.error(f"Inference failed: {e}")
            logger.exception("Inference error")

st.markdown("---")
st.markdown("**Model files expected at:**")
st.write(str(MODEL_DIR / f"encoder_model_{direction}.keras"))
st.write(str(MODEL_DIR / f"decoder_model_{direction}.keras"))
st.write(str(MODEL_DIR / f"token_tool_{direction}.pkl"))
st.write(str(MODEL_DIR / _spm_filename(direction)))