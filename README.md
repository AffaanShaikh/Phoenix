# Phoenix: The English-to-German Language translation model

**Live [here](https://phoenix-4rsr5nzdjknz3sansqta7c.streamlit.app/)**

### ver. 1.3.0
- switched to subword tokenization (SentencePiece) with learned embeddings and attention, improving decoded seq. length, enabling distributed subword representations, and letting the decoder focus on relevant encoder positions. 

- added LR warmup via WarmUpLRCallback: linearly ramp LR from warmup_initial_lr to peak_lr over warmup_epochs, compiled at the warmup start LR so ReduceLROnPlateau can take over after warmup without conflict.

- split training vs. evaluation: eval_only now loads the persisted token_tool and encodes the test set directly (no redundant SPM training/tokenization), ensuring consistency with saved models.

- preload the SentencePiece processor once for decoding to eliminate O(N) per-sample disk reads during eval (now, single file load: O(1)).

- reusing layers during inference correctly: fixed inference graph construction by creating fresh Input tensors and calling layer objects (sharing weights) rather than reusing training graph tensors, removing Keras graph-ambiguity failures.

- interface normalization: serialized Keras tokenizer as JSON ("keras_tokenizer_json") instead of pickling raw Tokenizer objects to ensure cross-version compatibility and reliable restore.

- streamlit app updated to support both char-level and subword-level tokenization and to present a normalized tokenizer interface for inference.

- miscellaneous:
    - eliminated data leakage, split raw texts (seed=42, test_ratio=0.2) before tokenization and train SPM exclusively on the training partition, removed post-tokenization index splitting.
    - exposed module hyperparameters to CLI 
    - replaced one-hot targets with SparseCategoricalCrossentropy to cut memory use.


### ver. 1.2.3
- added beam search decoding replacing bad performance amplifying greedy search 
- added bidirectional translation, dropout layers and adaptive learning rate


### ver. 1.2.0
- model retrained using dropout, better hyperparameters and regularization
- suffered from exposure bias i.e. stronger decoder compared to encoder


## Project overview

The project involves training a deep learning model for language translation from English to German (and vice-versa) using a sequence-to-sequence (seq2seq) architecture with LSTM layers and much more. The workflow spans from dataset prep., defining seq2seq model architecture and generating inference models capable of inference via HFHub deployed Streamlit app.

## Translations & Evaluation

model trained for 100 epochs (13 w/ EarlyStopping) on a subset of 50k samples (out of ~277k), evaluated was done on a held-out batch. 

**Translation direction:** DEU -> ENG

[Format: ver. 1.3.0 (ver. 1.2.3) -> improved by]
- **Exact match accuracy: 0.2774** (0.0257) 

    -> improved by **+0.2517 absolute / +979% relative**
- **Average CER (Character Error Rate): 0.3153** (0.4866)  
    -> improved by **−0.1713 absolute / 35.2% error reduction**

- **Average prediction length: 16.45** (16.18)  
    -> improved by **+0.27**

- **Average reference length: 16.93** (16.93)  
    -> **no change**

- **Prediction–reference gap: 0.48** (0.75)  
    -> improved by **−0.27 / 36% closer to reference**

- **Truncation count: 0** (12)  
    -> improved by **−12 (fully eliminated)**
 
These metrics reflect performance on the held-out 20% test split.

## Predictions

*Deutsch zu Englisch*
| Source (Input) | Reference | Model Prediction |
|----------------|-----------|------------------|
| Die Rechnung, bitte. | The check, please. | The bill, please. |
| Trinkt etwas Wasser. | Drink some water. | Drink some water. |
| Ich gieße Tee auf. | I'll make some tea. | I tead of them. |
| Suchen Sie sich jemand anders. | Find somebody else. | Have some someone. |
| Alle belügen mich. | Everyone lies to me. | Everybody twinked me. |
| Zieh Leine! | Go away! | Just get lost. |
| Ich bin arbeitslos. | I am out of work. | I'm unemployed. |
| Meine Uhr ist defekt. | My watch is broken. | My watch is finced. |
| Na gut! | That's fair enough. | That's good. |
| Wann fängt es an? | When does it begin? | When will it begin? |

*English to German* 
| Source (Input) | Reference | Model Prediction |
|----------------|-----------|------------------|
| Everyone likes her. | Alle haben sie gern. | Alle mögen sie. |
| Were you sick? | Waren Sie krank? | Warst du krank? |
| He shined his shoes. | Er wienerte seine Schuhe. | Er putzte seine Schuhe. |
| You are blushing. | Du wirst rot! | Du bist kämpfen! |
| We're still young. | Wir sind noch jung. | Wir sind immer noch jung. |
| This is my world. | Das ist meine Welt. | Das ist meine CD. |
| How are you feeling? | Wie fühlen Sie sich? | Wie geht es dir? |
| I'm a beginner. | Ich bin Anfängerin. | Ich bin Professor. |
| Did you bring yours? | Hast du deins mitgebracht? | Haben Sie Ihren mitgebracht? |
| I used to love that. | Sonst hat mir das immer sehr gefallen. | Ich habe das genossen. |

## Observations 
1. Improvements with sentencepiece + attention: exact-match rose from 2.57% -> 27.74% and CER dropped 0.4866 -> 0.3153. Truncations were fully eliminated, and prediction lengths are now closer to references, showing the new tokenization and decoding pipeline fixes alignment and cutoff issues.

2. **Fluency improved but semantic/lexical errors remain.** Many translations are natural and correct, yet there are still nonsensical outputs and meaning shifts (e.g., *Anfängerin* -> “Professor”), indicating surface-level generation improved but rare-word and lexical selection problems persist.

3. **Remaining challenges and evaluation limits.** Pronoun/formality mismatches (EN->DE) and semantic drift occur, and metrics are based on a small 50k subset with early stopping, so results may vary on the full dataset or other test conditions; further evaluation is needed.


## Data

Find the dataset at *http://www.manythings.org/anki* : or [download](http://www.manythings.org/anki/deu-eng.zip) the deu-eng.zip I used.  

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- https://keras.io/
- https://keras.io/2.16/api/layers/recurrent_layers/lstm/
- https://arxiv.org/abs/1409.3215
- https://sh-tsang.medium.com/review-seq2seq-sequence-to-sequence-learning-with-neural-networks-bcb84071a670
