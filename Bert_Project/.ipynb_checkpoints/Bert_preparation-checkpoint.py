import pickle
import os 
import re
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

Data_input = './Data_output/inputdata_title_content'
training_set = ArticleProcessor(Data_input).get_train_examples()
valid_set = ArticleProcessor(Data_input).get_dev_examples()

tokenizer = chinese_tokenizer()





