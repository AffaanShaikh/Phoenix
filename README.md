# Developing Machine Translation Systems: English-German Language translation

## Project overview

The project involves a deep learning model for machine translation from English to German using a sequence-to-sequence (seq2seq) architecture with LSTM layers. The workflow spans from data preprocessing to model training and inference. Initially, the dataset is prepared by extracting characters and organizing them into one-hot vectors. This step includes both input and target texts necessary for training. The seq2seq model architecture is defined next, comprising an LSTM-based encoder-decoder structure. The encoder processes input sequences, while the decoder predicts output sequences. During training, the model is optimized using categorical crossentropy loss and RMSprop optimizer, with accuracy as the metric. Validation data ensures model generalization. Post-training, inference models are generated from the trained model to translate new input sequences. Finally, trained weights and necessary indices are saved for future inference. This comprehensive approach covers all stages from initial data preparation to deploying a functional translation model.


## Translations:

After training the models for 100 epochs with a batch size of 64 and latent dimension of 256 on only 10000 samples (out of 277000~), the model was able to decode the characters and starting words accurately. However due to lower training time and being trained on only 10000 the model is not as accurate as one would like. Some of the translations I was able to get with the above configuration are below:-

- Input sentence:
    **She is happy.**

    Decoded sentence:
    *Sie ist mit under du binn.*

- Input sentence:
    **We have money.**

    Decoded sentence:
    *Wir haben Leben sichen.*

- Input sentence:
    **That's a joke.**

    Decoded sentence:
   *Das ist ein Witz.*

- Input sentence:
    **We need music.**

    Decoded sentence:
    *Wir brauchen Sie.*

- Input sentence:
    **I'm an officer.**

    Decoded sentence:
    *Ich bin ein Boftig.*

- Input sentence:
    **I feel blue.**

    Decoded sentence:
    *Ich fühle mich wehlig.*

- Input sentence:
    **Stay close.**

    Decoded sentence:
    *Bleiben Sie gesanden!*

- Input sentence:
    **Can you skip?**

    Decoded sentence:
    *Kannst du schwammen?*


## Project Structure

For the time being I'd like you to focus on `src\components`, rest of the repository is built to support further development as an application.

Within src/components/ you will find:
- `model_utils.py`: Contains functions for data processing, model definition, training, and decoding.
- `model_trainer.py`: Script that orchestrates the workflow by calling functions from `model_utils.py` to train and save the model.
- `model_prediction.py`: Script for using a trained model to translate English sentences to German.

The **logs** for all the training and the **saved models** can be found at:
 - `src/components/logs/fit` and `src/components/trained_models` respectively.

## Setup

To set up the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies:**
    - Note if you want to work with a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## Data

Please find the dataset for english to german translation at: `src/dataset/training_dataset/deu.txt` or download using: *http://www.manythings.org/anki/deu-eng.zip*

The dataset used for training and evaluation should be placed in a directory `data/` or specified as per your configuration in `model_utils.py`.

## Training the Model

To train the model, run:
```bash
cd src/components/ # if necessary
python model_trainer.py
```
This script will handle data preprocessing, model training, and save the trained model weights.

**Note**: On training, the encodings and the saved trained models will be replaced. Logs will be kept and updated.  

## Making Predictions

To use the trained model for inference, run:
```bash
python model_prediction.py --input_sentence "enter-here"
```
Replace `"enter-here"` with the actual sentence you want to translate.

## Next Steps

- Implement a Django or Flask application to provide a user interface for translation.
- Enhance the model by experimenting with different architectures or hyperparameters.
- Evaluate the model's performance on various metrics such as BLEU score.
- Perform additional training with different hyperparameters to further improve translation quality.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- https://keras.io/
- https://keras.io/2.16/api/layers/recurrent_layers/lstm/
- https://arxiv.org/abs/1409.3215
- https://sh-tsang.medium.com/review-seq2seq-sequence-to-sequence-learning-with-neural-networks-bcb84071a670