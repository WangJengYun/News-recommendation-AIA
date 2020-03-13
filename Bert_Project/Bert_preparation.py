import pickle
import tensorflow as tf
import re
import os 
os.chdir('/home/jovyan/at081-group39/python_code/Bert_Project/')
from Data_preparation_function import loading_pickle_file,Bert_Inputdata,finding_all_class,split_dataset

########## Data preparation ##########
all_news_path = '../../data_processed/news/all_news.p'
news_dict = loading_pickle_file(all_news_path)

Bertinputdata_1 = Bert_Inputdata(news_dict,text_type = 'title',method = 'single_sentence').Bert_Inputdata()
Bertinputdata_2 = Bert_Inputdata(news_dict,text_type = 'content',method = 'single_sentence').Bert_Inputdata()
Bertinputdata_3 = Bert_Inputdata(news_dict,text_type = 'content',method = 'pair_sentence').Bert_Inputdata()

# saving prepared_data
split_dataset(input_data = Bertinputdata_1,percent = 0.2,saving_file_name = 'inputdata_title')
split_dataset(input_data = Bertinputdata_2,percent = 0.2,saving_file_name = 'inputdata_content')
split_dataset(input_data = Bertinputdata_3,percent = 0.2,saving_file_name = 'inputdata_title_content')
# Bertinputdata_3[0]
########## BertData preparation ##########
import os 
os.chdir('/home/jovyan/at081-group39/python_code/Bert_Project/')
from Bert_preparation_function import ArticleProcessor , convert_single_example
from Bert_preparation_function import  chinese_tokenizer , file_based_convert_examples_to_features

tokenizer = chinese_tokenizer()

Data_input = './Data_output/inputdata_title_content'
training_set = ArticleProcessor(Data_input).get_train_examples()
valid_set = ArticleProcessor(Data_input).get_dev_examples()

file_based_convert_examples_to_features(training_set, max_seq_length = 128,tokenizer = tokenizer,
                                        file_name = 'title_content_word_train',handling_method = 'word')
file_based_convert_examples_to_features(valid_set, max_seq_length = 128,tokenizer = tokenizer, 
                                        file_name = 'title_content_word_valid',handling_method = 'word')

#===============================================
import tensorflow as tf
import os 
os.chdir('/home/jovyan/at081-group39/python_code/Bert_Project/')
from Bert_preparation_function import file_based_input_fn_builder

input_file = './Bert_TFrecord/title_content_word_valid.tf_record'

train_input_fn = file_based_input_fn_builder(input_file, seq_length = 128, is_training = True, drop_remainder = True)

from bert import modeling
from bert import optimization
import tensorflow as tf
from datetime import datetime
Bert_output = './Bert_output' ; os.makedirs(Bert_output,exist_ok= True)
BERT_CONFIG = './chinese_L-12_H-768_A-12/bert_config.json'
BERT_INIT_CHKPNT = './chinese_L-12_H-768_A-12/bert_model.ckpt'
bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)

BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1.0
WARMUP_PROPORTION = 0.1
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500

num_train_steps = int(48503 / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

run_config = tf.estimator.RunConfig(
    model_dir=Bert_output,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    keep_checkpoint_max=1,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = model_fn_builder(
  bert_config=bert_config,
  num_labels= 895,
  init_checkpoint=BERT_INIT_CHKPNT,
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps,
  use_tpu=False,
  use_one_hot_embeddings=False)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})


print(f'Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)
