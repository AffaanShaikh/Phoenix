import keras 
import pickle
import datetime
import numpy as np

# Batch size: Size of samples to train the model on.
batch_size = 64 #128

# Number of epochs to train for.
epochs = 100

# Latent dimensionality of the encoding space.
latent_dim = 256 #512 #1024  

# Number of samples to train on.
set_num_samples = 10000 #max: 277891

encoder_path='X:/LT-seq2seq/src/components/trained_models/encoder_model.keras'
decoder_path='X:/LT-seq2seq/src/components/trained_models/decoder_model.keras'

# Path to the dataset .txt file on disk.
data_path = "X:/LT-seq2seq/src/dataset/training_dataset/deu.txt"


def prepareData(data_path):
    """
    Prepares data for training a sequence-to-sequence model by extracting characters, 
    encoding them into one-hot vectors, and organizing necessary data structures.
    """
    input_characters, target_characters, input_texts, target_texts=extractChar(data_path, exchangeLanguage=False, num_samples=set_num_samples)
    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length =encodingChar(input_characters, target_characters, input_texts, target_texts) # extra num_decoder_tokens removed
    
    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, input_texts, target_texts, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length # extra num_decoder_tokens removed

def extractChar(data_path, exchangeLanguage=False, num_samples=None):
    """
    Extracts characters and texts from a language translation dataset.
    """
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
    print('input_characters:', input_characters)
    print('target_characters:', target_characters)
    print('input_texts:', input_texts[:100])
    print('target_texts:', target_texts[:100])
    
    return input_characters, target_characters, input_texts, target_texts

def encodingChar(input_characters,target_characters,input_texts,target_texts):
    """
    Encodes input and target texts into one-hot vectors for training a sequence-to-sequence model.
    """
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
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        # Pad encoder input sequences with the padding token
        encoder_input_data[i, len(input_text):, input_token_index[" "]] = 1.0
        #encoder_input_data[i, t + 1:, input_token_index[" "]] = 1.0 ##
        
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        # Pad decoder input and target sequences with the padding token
        decoder_input_data[i, len(target_text):, target_token_index[" "]] = 1.0
        decoder_target_data[i, len(target_text):, target_token_index[" "]] = 1.0
        #decoder_input_data[i, t + 1:, target_token_index[" "]] = 1.0 ##
        #decoder_target_data[i, t:, target_token_index[" "]] = 1.0 ##

    #print('encoder_input_data:', encoder_input_data)
    #print('decoder_target_data:', decoder_target_data)
    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length # extra num_decoder_tokens removed

def modelTranslation(num_encoder_tokens,num_decoder_tokens):
    """
    Defines a sequence-to-sequence translation model architecture using LSTM layers.
    """
    # Model arcitechture: 1 encoder(lstm) + 1 decode (LSTM) + 1 Dense layer + softmax
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens)) # Encoder Input layer,'None' represents the variable length of sequences.
    encoder = keras.layers.LSTM(latent_dim, return_state=True) # LSTM layer with latent_dim units
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens)) # Decoder Input layer
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True) # Another LSTM layer with the same number of units as the encoder.
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax') # a Dense layer with num_decoder_tokens units and a softmax activation func.
    decoder_outputs = decoder_dense(decoder_outputs) # final output of the decoder after applying the dense layer.

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs) # the Keras Model, which takes encoder and decoder inputs and produces the decoder outputs.
    
    return model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense

def trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data):
    """
    Trains, logs and saves the sequence-to-sequence model for translation.
    """

    log_dir = "X:/LT-seq2seq/src/components/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit([encoder_input_data, decoder_input_data],
              decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks = [tboard_callback])

    model.save("X:/LT-seq2seq/src/components/trained_models/sq2sq_model.keras")
    
def generateInferenceModel(input_token_index, target_token_index):
    """
    Generates inference models for sequence-to-sequence translation using a pretrained model.
    """
    # Restore the model and construct (sampling models) the encoder and decoder.
    model = keras.models.load_model("X:/LT-seq2seq/src/components/trained_models/sq2sq_model.keras")

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = keras.Input(shape=(latent_dim,))
    decoder_state_input_c = keras.Input(shape=(latent_dim,))
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

    return encoder_model,decoder_model,reverse_target_char_index

def saveChar2encoding(filename, input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index):
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
    with open(filename, "wb") as f:
        pickle.dump(input_token_index, f)
        pickle.dump(max_encoder_seq_length, f)
        pickle.dump(num_encoder_tokens, f)
        pickle.dump(reverse_target_char_index, f)
        pickle.dump(num_decoder_tokens, f)
        pickle.dump(target_token_index, f)

def decode_sequence(input_seq, encoder_model, decoder_model, num_decoder_tokens, target_token_index, reverse_target_char_index, max_decoder_seq_length_given):
    """
    Decode a sequence from the input sequence using the trained encoder and decoder models.
    """ 
    # encode the input.
    states_value = encoder_model.predict(input_seq, verbose = 0)

    # Generate empty target sequence of length 1.0
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.0

    stop_condition = False
    decoded_sentence = ""
    # predict the output letter by letter 
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose = 0) #, verbose = 0 for silent, 1 for bar, 2 for single line

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

    return decoded_sentence

def encodingSentenceToPredict(sentence, input_token_index, max_encoder_seq_length, num_encoder_tokens):
    """
    Encode a given sentence into its corresponding one-hot encoding based on the provided token index.
    """
    # Initialize the encoder input data with zeros.
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    
    for t, char in enumerate(sentence):
        if t < max_encoder_seq_length:  # Ensure we don't exceed the max length.
            if char in input_token_index:  # Ensure the character is in the token index
                encoder_input_data[0, t, input_token_index[char]] = 1.0
    
    return encoder_input_data

def getChar2encoding(filename):
    """
    Retrieve trained weights and indices needed for inference from the pickle file.
    """
    with open(filename, "rb") as f:
        input_token_index = pickle.load(f)
        max_encoder_seq_length = pickle.load(f)
        num_encoder_tokens = pickle.load(f)
        reverse_target_char_index = pickle.load(f)
        num_decoder_tokens = pickle.load(f)
        target_token_index = pickle.load(f)
        
    return input_token_index, max_encoder_seq_length, num_encoder_tokens, reverse_target_char_index, num_decoder_tokens, target_token_index