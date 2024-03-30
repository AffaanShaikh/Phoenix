from model_utils import *

# - train model
# - push saved model pickle file to cloud

# load the data, format it for processing
encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length=prepareData(data_path)

# build the model
model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense=modelTranslation(num_encoder_tokens,num_decoder_tokens)

# we train it
trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data)

# build the final model for the inference (and save it)
encoder_model,decoder_model,reverse_target_char_index=generateInferenceModel(encoder_inputs, encoder_states,input_token_index,target_token_index,decoder_lstm,decoder_inputs,decoder_dense)

# Save the object to convert: sequence to encoding and encoding to sequence
saveChar2encoding(r"X:\LT-seq2seq\src\components\output\char2encoding.pkl",input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index)