import pickle
import re
import random
import os 

def loading_pickle_file(path):
    with open(path,'rb') as file:
        data_dict = pickle.load(file)
    return data_dict

def finding_all_class(data_list):

    def customized_order(text):
        numeric_value = re.split(r'(\d+)', text)[1]
        return int(numeric_value) if numeric_value.isdigit() else numeric_value

    all_class_set = set()
    for index in range(len(data_list)):
        all_class_set = all_class_set.union(set(data_list[index]['mclass']))
    all_class_list = list(all_class_set)
    all_class_list.sort(key = customized_order)

    return all_class_list

class Bert_Inputdata(object):
    def __init__(self,news_dict_data,text_type,method,mapping_table = None):
        self.news_dict_data = news_dict_data
        self.text_type = text_type
        self.method = method
        self.mapping_table =mapping_table    
        self.all_class_list = finding_all_class(news_dict_data)
        
    def define_target(self,onenews_class):
        labels = map(lambda x : 1 if x in onenews_class  else 0,self.all_class_list)
        return list(labels)

    def Bert_Inputdata(self): 
        if self.method == 'single_sentence':
            example_size = len(self.news_dict_data)
            Bert_data = []
            for index in range(example_size):
                # index  = 1
                onenews_dict_data = self.news_dict_data[index]
                target = self.define_target(onenews_dict_data['class'])
                text_a = onenews_dict_data[self.text_type]
                text_b = None
                Bert_data.append((index,text_a,text_b,target))
            return Bert_data
        elif self.method == 'pair_sentence':
            example_size = len(self.news_dict_data)
            Bert_data = []
            for index in range(example_size):
                # index  = 1
                onenews_dict_data = self.news_dict_data[index]
                target = self.define_target(onenews_dict_data['class'])
                text_a = onenews_dict_data['title']
                text_b = onenews_dict_data['content']
                Bert_data.append((index,text_a,text_b,target))
            return Bert_data
#input_data = Bertinputdata_3;percent = 0.2;saving_file_name = 'inputdata_title_content'
def split_dataset(input_data,percent,saving_file_name):
    
    
    sample_size = len(input_data)
    index = list(range(0,sample_size))
    random.shuffle(index)
    
    validation_index = random.sample(index,int(sample_size*percent))
    training_index = list(filter(lambda x : x not in validation_index , index))

    validation_set = list(map(lambda x : input_data[x] , validation_index))
    training_set = list(map(lambda x : input_data[x] , training_index))

    folder_path  = os.path.join('./Data_output/',saving_file_name)
    os.makedirs(folder_path,exist_ok=True)
    training_file_path  = os.path.join(folder_path,saving_file_name + '_train.p')
    validation_file_path = os.path.join(folder_path,saving_file_name + '_valid.p')

    with open(training_file_path,'wb') as file:
        pickle.dump(training_set,file)

    with open(validation_file_path,'wb') as file:
        pickle.dump(validation_set,file)

if __name__ == "__main__":
    pass