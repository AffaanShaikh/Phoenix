import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
from keras.callbacks import TensorBoard
import pickle

# functions:
# - preparing and extracting data
# - training and saving model
# - encoder and decoder functions

# Latent dimensionality of the encoding space.
latent_dim = 1024#256  

# Batch size for training.
batch_size = 128

# Number of epochs to train for.
epochs = 2  

# Number of samples to train on.
num_samples = 10000 

encoder_path='encoder_modelPredTranslation.h5'
decoder_path='decoder_modelPredTranslation.h5'

# Path to the dataset ((.txt) file on disk.
data_path = r"X:\LT-seq2seq\src\dataset\training_dataset\ara.txt" #'ara.txt' 


# Data processing 
def prepareData(data_path):
    input_characters,target_characters,input_texts,target_texts=extractChar(data_path)
    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length =encodingChar(input_characters,target_characters,input_texts,target_texts)
    
    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length

def extractChar(data_path, exchangeLanguage=False, num_samples=None):
    # extract data from the dataset with each char (sentence1 ... sentence2)
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    if num_samples is None:
        num_samples = 100

    with open(data_path, encoding='utf-8') as f:
        lines = f.read().split('\n')
        print(str(len(lines) - 1))

        if exchangeLanguage == False:
            for line in lines[: min(num_samples, len(lines) - 1)]:
                parts = line.split('\t', maxsplit=1)
                if len(parts) == 2:
                    input_text, target_text = parts
                    target_text = '\t' + target_text + '\n'
                    input_texts.append(input_text)
                    target_texts.append(target_text)
                    input_characters.update(input_text)
                    target_characters.update(target_text)

        else:
            for line in lines[: min(num_samples, len(lines) - 1)]:
                parts = line.split('\t', maxsplit=1)
                if len(parts) == 2:
                    target_text, input_text = parts
                    target_text = '\t' + target_text + '\n'
                    input_texts.append(input_text)
                    target_texts.append(target_text)
                    input_characters.update(input_text)
                    target_characters.update(target_text)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))

    return input_characters, target_characters, input_texts, target_texts

def encodingChar(input_characters,target_characters,input_texts,target_texts):
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    print('Number of num_encoder_tokens:', num_encoder_tokens) 
    print('Number of samples:', len(input_texts))

    # encoder tokens: 26 uppercase letters + 26 lowercase letters + 19 punctuations and symbols = 71 encoder tokens
    print('Number of unique input tokens:', num_encoder_tokens) 

    # decoder tokens: 28 arabic alphabets + 71 all encoder + 43 numbers? bullshit like '\t'
    print('Number of unique output tokens:', num_decoder_tokens) 

    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.


    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length
	
# Model definition
def modelTranslation(num_encoder_tokens,num_decoder_tokens):
# Model arcitechture: 1 encoder(lstm) + 1 decode (LSTM) + 1 Dense layer + softmax

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    return model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense

# Model trainer
def trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data):

    LOG_PATH=r"X:\LT-seq2seq\src\components\output\log" 
        
    tbCallBack = TensorBoard(log_dir=LOG_PATH, histogram_freq=0, write_graph=True, write_images=True)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.01,
            callbacks = [tbCallBack])
    
# Inference model
def generateInferenceModel(encoder_inputs, encoder_states,input_token_index,target_token_index,decoder_lstm,decoder_inputs,decoder_dense):
# Connect the encoder/decoder and we create a new model and save
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)

    # Reverse-lookup token index to decode sequences back to something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())
    encoder_model.save(encoder_path)
    decoder_model.save(decoder_path)
    return encoder_model,decoder_model,reverse_target_char_index

# function to save trained weights 
def saveChar2encoding(filename,input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index):

    f = open(filename, "wb")
    pickle.dump(input_token_index, f)
    pickle.dump(max_encoder_seq_length, f)
    pickle.dump(num_encoder_tokens, f)
    pickle.dump(reverse_target_char_index, f)
    
    pickle.dump(num_decoder_tokens, f)
    
    pickle.dump(target_token_index, f)
    f.close()
