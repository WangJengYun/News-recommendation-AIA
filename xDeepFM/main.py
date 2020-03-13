import os 
import pandas as pd
import numpy as np
#reading news 
news_path = '/home/jovyan/at081-group39/data_processed/news/all_news_mcalss.p'
news_data = pd.DataFrame(reading_picke_file(news_path))
print('news_data = {}'.format(news_data.shape))

#####handling the news######
# mapping table 
_,news_mapping_table = laber_encoding(news_data['new_id'])

# handing mclass
mclass_list = news_data['mclass'].tolist()
padding_mclass,_ = mclass_encoding(mclass_list)
mclass_df = pd.DataFrame(padding_mclass,index = news_data['new_id'])
print('mclass_df = {}'.format(mclass_df.shape))

# Bert_vector 
bert_vector_path = '/home/jovyan/at081-group39/python_code/Bert_Project/output_title.jsonl'
Bert_df = read_Bert_vector(bert_vector_path)
print('Bert_df = {}'.format(Bert_df.shape))
######################################
######################################
#####user log######
# reading log data 
log_path = '/home/jovyan/at081-group39/python_code/1_deepctr_models/model-data.pkl'
log_data = reading_picke_file(log_path) 
print('log_data = {}'.format(log_data.shape))
user_count = len(log_data['user_id'].unique())

# 選取資料大小
user_total_read = 50
log_data = log_data[log_data['total_read'] >= user_total_read]
print('log_data = {}'.format(log_data.shape))
log_data['user_id'].nunique()

user_id_filter = log_data['user_id']

# dropna
log_data = log_data.dropna()
print('log_data = {}'.format(log_data.shape))

# label encoding for user_data
from sklearn.preprocessing import LabelEncoder
Labelmodel_dict = dict()
cat_fest  = ['user_id','config_browser_name','config_os']
for feat in cat_fest:
    Labelmodel = LabelEncoder()
    Labelmodel.fit(log_data[feat])
    Labelmodel_dict[feat] = Labelmodel
    log_data[feat] = Labelmodel.transform(log_data[feat])

# new_id label encodeing
news_lbe  = LabelEncoder()
news_lbe.fit(log_data['news_id'])

## 選取資料大小
#user_total_read = 100
#log_data = log_data[log_data['total_read'] >= user_total_read]
#print('log_data = {}'.format(log_data.shape))
#log_data['user_id'].nunique()

#####label =1
log_data['label'] = 1

# split data into train data and test data
log_data['time'] = pd.to_datetime(log_data['time'])
date_cut=pd.to_datetime("20181001 00:00:00")
train_set = log_data.query('time < @date_cut')
print('train_set = {}'.format(train_set.shape))
test_set = log_data.query('time >= @date_cut')
print('test_set = {}'.format(test_set.shape))

#----------------------
user_feature_names = ['user_id','config_browser_name','config_os','time']
news_feaure_names = ['news_id','create_date']
train_set = generate_negative_sammple(train_set,user_feature_names,news_feaure_names)
print('train_set = {}'.format(train_set.shape))
#----------------------
from sklearn.utils import shuffle
train_set = shuffle(train_set)

train_bert = Bert_df.loc[train_set['news_id'].tolist(),:]
train_mclass = mclass_df.loc[train_set['news_id'].tolist(),:]
print('train_bert = {}'.format(train_bert.shape))
print('train_mclass = {}'.format(train_mclass.shape))

test_bert = Bert_df.loc[test_set['news_id'].tolist(),:]
test_mclass = mclass_df.loc[test_set['news_id'].tolist(),:]
print('test_bert = {}'.format(test_bert.shape))
print('test_mclass = {}'.format(test_mclass.shape))

# input data to model
train_input,mms = relative_news_handing(train_set,handle_mms_name = ['sub_time'],news_lbe = news_lbe,mms = None)
test_input,_ = relative_news_handing(test_set.copy(),handle_mms_name = ['sub_time'] ,news_lbe = news_lbe,mms= mms)

#len(train_set['user_id'].unique())
#len(test_set['user_id'].unique())
######## evaluate data ######
#predict_path = '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_2000.pkl'
#predict_data = reading_picke_file(predict_path) 
#print('pred_data = {}'.format(predict_data .shape))
#predict_data = predict_data[predict_data['user_id'].isin(user_id_filter)]
#print('pred_data = {}'.format(predict_data .shape))
#len(predict_data['user_id'].unique())
#
## label encoding for user_data
#cat_fest  = ['user_id','config_browser_name','config_os']
#for feat in cat_fest:
#    predict_data[feat] = Labelmodel_dict[feat].transform(predict_data[feat])
#
#predict_bert = Bert_df.loc[predict_data['news_id'].tolist(),:]
#predict_mclass = mclass_df.loc[predict_data['news_id'].tolist(),:]
#print('predict_bert = {}'.format(predict_bert.shape))
#print('predict_mclass = {}'.format(predict_mclass.shape))
#
#predict_input,_ = relative_news_handing(predict_data,handle_mms_name = ['sub_time'],news_lbe = news_lbe, mms= mms)
#print('predict_set = {}'.format(predict_input.shape))
#predict_input = predict_input[['user_id','news_id','config_browser_name','config_os','sub_time','label']]

'''
df = test_set.iloc[3467]
user_feature_names = ['user_id','config_browser_name','config_os','time']
news_feaure_names = ['news_id','create_date']
select_day = 3

predict_data_for_oneuser = generating_predict_data(df,log_data,user_feature_names,news_feaure_names,select_day)
predict_bert = Bert_df.loc[predict_data_for_oneuser['news_id'].tolist(),:]
predict_mclass = mclass_df.loc[predict_data_for_oneuser['news_id'].tolist(),:]
print('predict_bert = {}'.format(predict_bert.shape))
print('predict_mclass = {}'.format(predict_mclass.shape))
predict_input,_ = relative_news_handing(predict_data_for_oneuser,handle_mms_name = ['sub_time'],news_lbe = news_lbe, mms= mms)
print('predict_set = {}'.format(predict_input.shape))
'''
#-----------------------------------------------------
#-----------------------------------------------------
# transform datatype before training model
class feature_spec(object):
    def __init__(self,name,dimension,dtype,maxlen = 1):
        self.name = name
        self.dimension = dimension
        self.maxlen = maxlen
        self.dtype = dtype
        
cat_featname = ['user_id','news_id','config_browser_name','config_os']
cont_feature = ['sub_time']

feat = 'config_os'
cat_feat_list = [feature_spec(feat,len(log_data[feat].unique()),dtype = 'float32')\
    for feat in cat_featname]

#cat_featname = ['news_id','config_browser_name','config_os']
#cat_feat_list = [feature_spec('news_id',user_count,dtype = 'float32')] + [feature_spec(feat,len(log_data[feat].unique()),dtype = 'float32')\
#    for feat in cat_featname]

cont_feat_list = [feature_spec(feat,1,dtype = 'float32')\
    for feat in cont_feature]

bert_feat = feature_spec('bert',768,dtype = 'float32')

news_mclass = feature_spec('mclass',17,dtype = 'float32')

feat_dict= {'cat':cat_feat_list,'cont':cont_feat_list,'bert':bert_feat,'mclass':news_mclass}

#train_input_fn,train_target  = input_data_for_model(train_input,[],feat_dict)
#test_input_fn,test_target  = input_data_for_model(test_input,[],feat_dict)
train_input_fn,train_target  = input_data_for_model(train_input,train_bert,train_mclass,feat_dict)
test_input_fn,test_target  = input_data_for_model(test_input,test_bert,test_mclass,feat_dict)
#================================
#================================
#================================
from keras.layers import Input,Embedding,Dense,Concatenate,Reshape,Flatten
from keras.layers import BatchNormalization,Dropout,Activation,Dot,Add
from keras.regularizers import l2
from keras import losses
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.callbacks import EarlyStopping
from keras.models import Model
from collections import OrderedDict
#from tensorflow.keras.layers import Flatten

#-------
##### input_data for model #####
cat_input_dict,cont_input_dict,Bert_Input,mclass_Input,inputs_list = model_input(feat_dict)
##### embedding #####
embedding_size = 64;
#l2_reg_embedding=0.00001
#init_std = 0.0001
#seed =  124
#-------------------
Deep_input_list = embedding_layer(cat_input_dict,cont_input_dict,Bert_Input,mclass_Input,\
    embedding_size)
linear_output = linear_embedding_layer(cat_input_dict,cont_input_dict,Bert_Input,mclass_Input)
######################
#Flatten_list = []
#for item in Deep_input_list:
#    Flatten_list.append(Flatten()(item))
#DD_1 = Dot(1)([Flatten_list[0],Flatten_list[1]])
#DD_2 = Dot(1)([Flatten_list[2],Flatten_list[3]])
#DD_F = Dot(1)([DD_1,DD_2])
###### transform ######
Deep_input_1 = Concatenate(axis = 1)(Deep_input_list)
final_Deep_input = Flatten()(Deep_input_1)
##--------------
#DNN
dnn_hidden_units = (512,256,128)
input_layer = final_Deep_input
use_bn = True
is_traning = True
dropout_rate = 0.2
#------------
DNN_input = input_layer
for i in range(len(dnn_hidden_units)):
    fc = Dense(dnn_hidden_units[i], activation=Activation('relu'))\
        (DNN_input)
    
    if use_bn : 
        fc = BatchNormalization()(fc)
    
    fc = Dropout(dropout_rate)(fc)
    DNN_input = fc
Deep_output = DNN_input
###### FM ######
CIN_layer_size = (128,64)
#--------------------
FM_output = FM(Deep_input_1)
CIN_output = CIN(Deep_input_1,CIN_layer_size)
###### Concatenate ######
method = 'xDeepFM'
if method == 'DeepFM':
    all_part_output = Concatenate()([FM_output,Deep_output,linear_output])
elif method == 'xDeepFM':
    all_part_output = Concatenate()([CIN_output,Deep_output,linear_output])
else:
    all_part_output = Deep_output
###### output layer ######
#output = Dense(1,name = 'output')(all_part_output)
output = Dense(1,activation=Activation('sigmoid'),name = 'output')(all_part_output)
#output = Activation('sigmoid')(DD_1)
#output = DD_1
model = Model(inputs=inputs_list, outputs=output)
model.summary()
earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)

model.compile(loss = losses.binary_crossentropy,
                    optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

#model.compile(loss=losses.mean_squared_error,
#              optimizer=Adam(lr=1e-3),
#             metrics=['accuracy'])


history = model.fit(train_input_fn, train_target,
                    batch_size=256, epochs=1, verbose=1, 
                    validation_data=(test_input_fn,test_target), 
                    callbacks=[earlystop])

#y_predic = model.predict(test_input_fn)

#(y_predic>0.5).mean()


##### evaluate data ######
# path =  '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_50u5d.pkl'
def  ctr_evaluation(path):
    predict_data = reading_picke_file(path) 
    #print('pred_data = {}'.format(predict_data .shape))
    predict_data = predict_data[predict_data['user_id'].isin(user_id_filter)]
    print('user_count = {}'.format(len(predict_data['user_id'].unique())))

    # label encoding for user_data
    cat_fest  = ['user_id','config_browser_name','config_os']
    for feat in cat_fest:
        predict_data[feat] = Labelmodel_dict[feat].transform(predict_data[feat])

    predict_bert = Bert_df.loc[predict_data['news_id'].tolist(),:]
    predict_mclass = mclass_df.loc[predict_data['news_id'].tolist(),:]
    #print('predict_bert = {}'.format(predict_bert.shape))
    #print('predict_mclass = {}'.format(predict_mclass.shape))

    predict_input,_ = relative_news_handing(predict_data,handle_mms_name = ['sub_time'],news_lbe = news_lbe, mms= mms)
    #print('predict_set = {}'.format(predict_input.shape))
    predict_input = predict_input[['user_id','news_id','config_browser_name','config_os','sub_time','label']]


    predict_input_fn,target  = input_data_for_model(predict_input,predict_bert,predict_mclass,feat_dict,is_predict = True)
    pred_ans = model.predict(predict_input_fn, batch_size=256)

    predict_input['scores'] = pred_ans
    hit_rate = []
    for top_K in [1,3,5,10,15,20]:

        result = predict_input.groupby(['user_id']).apply(lambda df : df.sort_values(by = 'scores',ascending=False).head(top_K))
        rank = result['label'].sum() / len(result['user_id'].unique())
        hit_rate.append(rank)
        #print('top_k = {} : {}'.format(top_K,rank))
    return hit_rate

result_df = pd.DataFrame(['TOP1','TOP3','TOP5','TOP10','TOP15','TOP20'],columns =['CTR']) 
predict_path_50 = '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_50u5d.pkl'
result_df['50u5d'] =  ctr_evaluation(predict_path_50)
predict_path_2000 = '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_2000u5d.pkl'
result_df['1201u5d'] =  ctr_evaluation(predict_path_2000)
predict_path_5000 = '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_5000u5d.pkl'
result_df['1676u5d'] =  ctr_evaluation(predict_path_5000)
predict_path_8000 = '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_8000u5d.pkl'
result_df['1843u5d'] =  ctr_evaluation(predict_path_8000)
predict_path_8000 = '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_29849u5d.pkl'
result_df['2258u5d'] =  ctr_evaluation(predict_path_8000)
result_df

predict_path_50 = '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_50u30d.pkl'
result_df['50u30d'] =  ctr_evaluation(predict_path_50)
predict_path_2000 = '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_2000u30d.pkl'
result_df['1201u30d'] =  ctr_evaluation(predict_path_2000)
predict_path_5000 = '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_5000u30d.pkl'
result_df['1676u30d'] =  ctr_evaluation(predict_path_5000)
predict_path_8000 = '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_8000u30d.pkl'
result_df['1843u30d'] =  ctr_evaluation(predict_path_8000)
predict_path_8000 = '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_29849u30d.pkl'
result_df['2258u30d'] =  ctr_evaluation(predict_path_8000)

result_df
#=================================================================
#=================================================================
path = '/home/jovyan/at081-group39/python_code/1_deepctr_models/eval_data_8000u5d.pkl'
predict_data = reading_picke_file(path) 
#print('pred_data = {}'.format(predict_data .shape))
predict_data = predict_data[predict_data['user_id'].isin(user_id_filter)]
print('user_count = {}'.format(len(predict_data['user_id'].unique())))
# label encoding for user_data
cat_fest  = ['user_id','config_browser_name','config_os']
for feat in cat_fest:
    predict_data[feat] = Labelmodel_dict[feat].transform(predict_data[feat])
predict_bert = Bert_df.loc[predict_data['news_id'].tolist(),:]
predict_mclass = mclass_df.loc[predict_data['news_id'].tolist(),:]
#print('predict_bert = {}'.format(predict_bert.shape))
#print('predict_mclass = {}'.format(predict_mclass.shape))
predict_input,_ = relative_news_handing(predict_data,handle_mms_name = ['sub_time'],news_lbe = news_lbe, mms= mms)
#print('predict_set = {}'.format(predict_input.shape))
predict_input = predict_input[['user_id','news_id','config_browser_name','config_os','sub_time','label']]
predict_input_fn,target  = input_data_for_model(predict_input,predict_bert,predict_mclass,feat_dict,is_predict = True)
pred_ans = model.predict(predict_input_fn, batch_size=256)
predict_input['scores'] = pred_ans

result = predict_input.groupby(['user_id']).apply(lambda df : df.sort_values(by = 'scores',ascending=False))
result['user_id'] = Labelmodel_dict['user_id'].inverse_transform(result['user_id'])
result['news_id'] = news_lbe.inverse_transform(result['news_id'])
result = result.reset_index(drop = True)

import pickle 
path = '/home/jovyan/at081-group39/data_processed/evaluation/result_8000u5d'
with open(path,'wb') as file:
    pickle.dump(result, file)

with open(path,'rb') as file:
    A = pickle.load(file)

