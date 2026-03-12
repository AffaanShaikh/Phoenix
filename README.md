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

The project involves a deep learning model for machine translation from English to German (and vice-versa) using a sequence-to-sequence (seq2seq) architecture with LSTM layers. The workflow spans from data preprocessing to model training and inference. Initially, the dataset is prepared by extracting characters and organizing them into one-hot vectors. This step includes both input and target texts necessary for training. The seq2seq model architecture is defined next, comprising an LSTM-based encoder-decoder structure. The encoder processes input sequences, while the decoder predicts output sequences.  Post-training, inference models are generated from the trained model to translate new input sequences. Trained weights and necessary indices are saved for future inference. The approach covers stages from data prep. to deploying a functional translation model on a streamlit app.


## Translations & Evaluation

After training the model for 100 epochs on a subset of 50k samples (out of ~277k), we evaluated translation quality on a held-out batch. 

* **Translation direction:** DEU -> ENG
* **Number of test samples:** 10,000
* **Exact match accuracy:** 0.0257
* **Average CER (Character Error Rate):** 0.4866
* **Average prediction length:** 16.18
* **Average reference length:** 16.93
* **Truncation count:** 12 

These metrics reflect performance on the held-out 20% test split.

### Predictions (*sq2sq_model_deu2eng.keras*)
Decoding done with beam search.

| Source (Input)                                   | Reference            | Model Prediction     |
| ------------------------------------------------ | -------------------- | -------------------- |
| Komm schnell!                                    | Come quick!          | Come quick.          |
| Wir sollten gehen.                               | We should go.        | We should go.        |
| Woraus ist es gemacht?                         | What's it made from?   | What's it work?   |
| Ich werde euch meine leihen.                     | I'll loan you mine.  | I'll lend you mine.  |
| Tom ist echt cool.                              | Tom is really cool. | Tom is really good.      |
| Habt ihr sie mitgebracht?                          | Did you bring it?  | Did you bring you?       |
| Runter vom Rasen!                                 | Get off the lawn. | Get off the car.  |
| Sind Sie zu Fuß nach Hause gegangen?                          | Did you walk home?   | Did you get home?   |
| Sie können mich nicht entlassen.                  | You can't fire me.    | You can't stop you.     |
| Seid ihr zu Fuß nach Hause gegangen?                                | Did you walk home?     | Did you get home OK?   |
| Ich kann jetzt nichts essen.                                 | I can't eat now.        | I can't do that.        |
| Lassen Sie das mich machen.                        | Let me do this.  | Let me help you.     |
| Ich möchte auch gehen.                        | I want to go, too. | I want to see you.      | 

### Observations
- The model generates ok translations for shorter sequences.
- Exact match accuracy remains low unfortunetly (approx. 2.6%), indicating semantic errors.
- Prediction lengths are close to reference lengths, suggesting reasonable decoding stability.
- CER approx. 0.49 shows moderate character-level divergence.
- Truncation events are rare (12 / 10,000), implying minimal sequence cutoff issues.
- To resolve bad longer predictions due to encoder's bottleneck we'll soon introduce attention.

## Data

Find the dataset at *http://www.manythings.org/anki* : or [download](http://www.manythings.org/anki/deu-eng.zip) the deu-eng.zip I used.  

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- https://keras.io/
- https://keras.io/2.16/api/layers/recurrent_layers/lstm/
- https://arxiv.org/abs/1409.3215
- https://sh-tsang.medium.com/review-seq2seq-sequence-to-sequence-learning-with-neural-networks-bcb84071a670
