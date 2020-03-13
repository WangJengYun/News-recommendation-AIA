import pickle
import tensorflow as tf
import re
import os 
os.chdir('/home/jovyan/at081-group39/python_code/Bert_Project/')
from Bert_preparation_function import  chinese_tokenizer
from bert import tokenization

class InputExample(object):

  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids

class Inputdata_for_FE(object):
    def __init__(self,news_dict_data,text_type,method):
        self.news_dict_data = news_dict_data
        self.text_type = text_type
        self.method = method
        
    def Bert_Inputdata(self): 
        
        if self.method == 'single_sentence':
            ##### test #####
#             example_size = 100  #len(self.news_dict_data)
            ##### test #####
            example_size = len(self.news_dict_data)
            examples = []
            unique_id = 0
            for index in range(example_size):
                # index  = 1
                onenews_dict_data = self.news_dict_data[index]
                text_a = onenews_dict_data[self.text_type]
                text_b = None
                examples.append(
                    InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
                unique_id += 1    
            return examples
        elif self.method == 'pair_sentence':
            ##### test #####
#             example_size = 2  #len(self.news_dict_data)
            ##### test #####
            example_size = len(self.news_dict_data)
            examples = []
            unique_id = 0 
            for index in range(example_size):
                # index  = 1
                onenews_dict_data = self.news_dict_data[index]
                text_a = onenews_dict_data['title']
                text_b = onenews_dict_data['content']
                examples.append(
                    InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
                unique_id += 1 
            return examples

def sentence_truncate_seq_pair(text_a_sentence, text_b_sentence, max_length,tokenizer):
  tokens_a_list = [tokenizer.tokenize(sentence) for sentence in text_a_sentence]
  tokens_b_list = [tokenizer.tokenize(sentence) for sentence in text_b_sentence]
  while True:
    tokens_a = [y for x in tokens_a_list for y in x]  
    tokens_b = [y for x in tokens_b_list for y in x]   
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a_list.pop()
    else:
      tokens_b_list.pop()
  return tokens_a ,tokens_b

# max_length = max_seq_length - 3
def word_truncate_seq_pair(tokens_a, tokens_b, max_length):
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def loading_pickle_file(path):
    with open(path,'rb') as file:
        data_dict = pickle.load(file)
    return data_dict

def convert_examples_to_features(examples, seq_length, tokenizer,handling_method = 'word'):

    features = []

    for (ex_index, example) in enumerate(examples):
        if handling_method == 'word':
          # word_handling
            text_a_sentence = example.text_a.split('<SEP>')
            text_a = ''.join(text_a_sentence)
            tokens_a = tokenizer.tokenize(text_a)     
            tokens_b = None
            if example.text_b :
                text_b_sentence = example.text_b.split('<SEP>')
                text_b = ''.join(text_b_sentence)
                tokens_b = tokenizer.tokenize(text_b)       
            if tokens_b:
                word_truncate_seq_pair(tokens_a, tokens_b, (seq_length - 3))
            else :
                if len(tokens_a) > (seq_length - 2):
                    tokens_a = tokens_a[0:(seq_length - 2)]
        elif handling_method == 'sentence':
            # sentence_handling
            text_a_sentence = example.text_a.split('<SEP>')
            text_b_sentence = None
            if example.text_b :
                text_b_sentence =  example.text_b.split('<SEP>')        
            tokens_b = None
            if text_b_sentence : 
                tokens_a ,tokens_b = sentence_truncate_seq_pair(text_a_sentence, text_b_sentence, (max_seq_length - 3),tokenizer)
            else :
                tokens_a_list = [tokenizer.tokenize(sentence) for sentence in text_a_sentence]
                total_num = 0
                truncate_sentence_tokens = []
                for sentence_tokens in tokens_a_list :
                    total_num = total_num + len(sentence_tokens)
                    if total_num <=  (max_seq_length - 2):
                        truncate_sentence_tokens.append(sentence_tokens)
                    else :
                        break       
                tokens_a = [y for x in truncate_sentence_tokens for y in x]  

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        segment_type = 0

        if tokens_b :
          tokens_list = [tokens_a,tokens_b]
        else :
          tokens_list = [tokens_a]

        for tokens_slice in tokens_list :
          for token in tokens_slice:
            tokens.append(token)
            input_type_ids.append(segment_type)
          tokens.append("[SEP]")
          input_type_ids.append(segment_type)
          segment_type = 1 if segment_type == 0 else 0

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < seq_length:
          input_ids.append(0)
          input_mask.append(0)
          input_type_ids.append(segment_type)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
          tf.logging.info("*** Example ***")
          tf.logging.info("unique_id: %s" % (example.unique_id))
          tf.logging.info("tokens: %s" % " ".join(
              [tokenization.printable_text(x) for x in tokens]))
          tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
          tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
          tf.logging.info(
              "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))

    return features

def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


if __name__ == "__main__":
    
    tokenizer = chinese_tokenizer()
    all_news_path = '../../data_processed/news/all_news.p'
    seq_length = 128
    news_dict = loading_pickle_file(all_news_path)
    Bertinputdata_1 = Inputdata_for_FE(news_dict,text_type = 'title',method = 'single_sentence').Bert_Inputdata()
    features = convert_examples_to_features(
      examples=Bertinputdata_1, seq_length=seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
      unique_id_to_feature[feature.unique_id] = feature

    input_fn = input_fn_builder(
      features=features, seq_length=seq_length)
  

    