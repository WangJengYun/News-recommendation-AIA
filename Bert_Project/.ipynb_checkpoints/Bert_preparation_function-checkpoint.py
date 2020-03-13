import os 
import pickle
from bert import tokenization
import tensorflow as tf
import csv
import collections

class InputExample(object):
    
    def __init__(self, guid, text_a, text_b=None, labels=None):

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self,data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()
	
  def get_dev_examples(self,data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()
	
  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()
  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()
	
  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

class ArticleProcessor(DataProcessor):
  def __init__(self,datapath):
    self.datapath = datapath
  def get_train_examples(self, data_dir = None):
    """See base class."""       
    file_name = list(filter(lambda x :x.find("train") != -1,os.listdir(self.datapath)))[0]
    with open(os.path.join(self.datapath, file_name), "rb") as pickle_file:
      bert_train = pickle.load(pickle_file)   
    return self._create_example(bert_train,'train')
  def get_dev_examples(self, data_dir = None):
    file_name = list(filter(lambda x :x.find("valid") != -1,os.listdir(self.datapath)))[0]
    with open(os.path.join(self.datapath, file_name), "rb") as pickle_file:
      bert_train = pickle.load(pickle_file)   
    return self._create_example(bert_train,'train')
  def get_test_examples(self, data_dir = None):
    """See base class."""
    return None   
  def get_labels(self):
    """See base class."""
    return [0,1]
  def _create_example(self,Bert_Inputdata,set_type):
    Bert_Data = []
    for oneBert_Inputdata in Bert_Inputdata:
      #index  = 0
      guid = "%s_%s" % (set_type,oneBert_Inputdata[0])
      text_a = oneBert_Inputdata[1]
      text_b = None if bool(oneBert_Inputdata[2]) == False else oneBert_Inputdata[2]
      labels = oneBert_Inputdata[3]
      Bert_Data.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, labels=labels))
    return Bert_Data


class InputFeatures(object):
  def __init__(self, input_ids, input_mask, segment_ids, label_ids, is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.is_real_example=is_real_example 

def chinese_tokenizer():
  
  BERT_INIT_CHKPNT = './chinese_L-12_H-768_A-12/bert_model.ckpt'
  BERT_VOCAB= './chinese_L-12_H-768_A-12/vocab.txt'

  tokenization.validate_case_matches_checkpoint(True,BERT_INIT_CHKPNT)
  tokenizer = tokenization.FullTokenizer(
        vocab_file=BERT_VOCAB, do_lower_case=True)
  return  tokenizer

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """  

# max_length = max_seq_length - 3
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

def convert_single_example(ex_index, example, max_seq_length,tokenizer,handling_method = 'word'):

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_ids=0,
        is_real_example=False)

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
      word_truncate_seq_pair(tokens_a, tokens_b, (max_seq_length - 3))
    else :
      if len(tokens_a) > (max_seq_length - 2):
        tokens_a = tokens_a[0:(max_seq_length - 2)]
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

  #=====================================
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  segment_type = 0

  if tokens_b :
    tokens_list = [tokens_a,tokens_b]
  else :
    tokens_list = [tokens_a]

  for tokens_slice in tokens_list :
    for token in tokens_slice:
      tokens.append(token)
      segment_ids.append(segment_type)
    tokens.append("[SEP]")
    segment_ids.append(segment_type)
    segment_type = 1 if segment_type == 0 else 0

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(segment_type)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  labels_ids = []
  for label in example.labels:
    labels_ids.append(int(label))

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: {}".format(labels_ids))


  feature = InputFeatures(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          label_ids=labels_ids,
          is_real_example=True)
          
  return feature
 # examples = valid_set; max_seq_length = 512;tokenizer = tokenizer;file_name = 'title_sentence_valid';handling_method = 'sentence'

def file_based_convert_examples_to_features(examples, max_seq_length, tokenizer, file_name,handling_method = 'word'):
  file_path = os.path.join('./Bert_TFrecord/',file_name +'.tf_record')
  with tf.python_io.TFRecordWriter(file_path) as writer : 
    for (ex_index,example) in enumerate(examples) : 
      # ex_index = 431
      #  (ex_index,example) = list(enumerate(examples))[ex_index]
      
      feature = convert_single_example(ex_index, example,max_seq_length,tokenizer,handling_method = handling_method)

      def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

      features = collections.OrderedDict()
      features["input_ids"] = create_int_feature(feature.input_ids)
      features["input_mask"] = create_int_feature(feature.input_mask)
      features["segment_ids"] = create_int_feature(feature.segment_ids)
      features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
      
      '''
      if isinstance(feature.label_ids, list):
        label_ids = feature.label_ids
      else:
        label_ids = feature.label_ids[0]
      '''

      features["label_ids"] = create_int_feature(feature.label_ids)

      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      tf_example.SerializeToString()
      writer.write(tf_example.SerializeToString())

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([101], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }
  def _decode_record(record, name_to_features):
      
    example = tf.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t
    return example
  def input_fn(params):
      
    batch_size = params["batch_size"]
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
    return d
  return input_fn


if __name__ == "__main__":
    pass