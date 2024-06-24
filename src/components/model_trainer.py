from model_utils import *

encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index, input_texts, target_texts, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length=prepareData(data_path) 

# Build the sequence-to-sequence model
model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense=modelTranslation(num_encoder_tokens,num_decoder_tokens)

# Train the sequence-to-sequence model and save as 'sq2sq_model.keras'
trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data)

# Build the encoder-decoder inference model and save it
encoder_model,decoder_model,reverse_target_char_index=generateInferenceModel(input_token_index,target_token_index)

#Save the character encodings in a pickle file, save object to convert: sequence to encoding and encoding to sequence
saveChar2encoding("char2encoding.pkl",input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index)