# LT-seq2seq Copilot Instructions

## Project Overview
This is a character-level English-to-German machine translation system using Keras seq2seq LSTM architecture. The model processes text at the character level with one-hot encoding, not word tokens.

## Architecture
- **Encoder-Decoder**: LSTM-based seq2seq with shared latent dimension (256)
- **Data Processing**: Character extraction → one-hot encoding → training on parallel English-German pairs
- **Training**: RMSprop optimizer, categorical crossentropy, TensorBoard logging
- **Inference**: Separate encoder/decoder models for prediction

## Key Components
- `src/components/model_utils.py`: Core functions for data prep, model building, training, inference
- `src/components/model_trainer.py`: Orchestrates full training pipeline
- `src/components/model_prediction.py`: CLI prediction interface (`python model_prediction.py --input_sentence "text"`)

## Conventions
- **Character-level processing**: All text handled as sequences of individual characters (unusual for translation)
- **One-hot encoding**: Fixed vocabulary sizes for input/output characters
- **Hardcoded paths**: Models saved to `src/components/trained_models/`, logs to `src/components/logs/fit/`
- **Batch training**: 64 batch size, 100 epochs, 10k samples from deu.txt dataset
- **Special tokens**: `\t` (start), `\n` (end) for target sequences

## Workflows
- **Training**: Run `python src/components/model_trainer.py` - generates encoder/decoder models + char2encoding.pkl
- **Prediction**: `python src/components/model_prediction.py --input_sentence "Hello world"`
- **Data**: Uses tab-separated English-German pairs from `src/dataset/training_dataset/deu.txt`

## Dependencies
- Keras/TensorFlow for model operations
- NumPy for data manipulation
- Pickle for serialization of token indices

## Gotchas
- Paths use absolute X:/ drive references - update for your environment
- Character-level translation produces imperfect but recognizable German output
- Inference requires both encoder_model.keras and decoder_model.keras plus char2encoding.pkl
- TensorBoard logs accumulate in timestamped directories</content>
<parameter name="filePath">c:\Arbeit\Phoenix\LT-seq2seq\.github\copilot-instructions.md