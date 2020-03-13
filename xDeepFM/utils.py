import pickle
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import datetime

# path = log_path 
def reading_picke_file(path):
    with open(path,'rb') as file :
        result = pickle.load(file)
    return  result

def  laber_encoding(feat):
    feat_encoding = LabelEncoder().fit_transform (feat)
    mapping_table = dict(zip(feat,feat_encoding))
    return feat_encoding,mapping_table

def mclass_encoding(mclass_list) :

    all_mclass = []
    mclass_len = []
    for row in  mclass_list :
        # row  = mclass_list[0]
        all_mclass.extend(row)
        mclass_len.append(len(row))

    all_mclass = list(set(all_mclass))
    _,mclass_mapping_table = laber_encoding(all_mclass)

    mclass_encoding = []
    for row in mclass_list:
        mclass_encoding.append([mclass_mapping_table[mclass] for mclass in row])

    result = pad_sequences(mclass_encoding, maxlen=max(mclass_len), padding='post', dtype=np.float, value=0)

    return result,mclass_mapping_table

def read_Bert_vector(bert_vector_path):
    with open(bert_vector_path) as json_file: 
        data = json_file.readlines()

    Bert_vector  = {}
    for lines in data :
        d = json.loads(lines)
        First_step = True
        for layer in ['-1','-2','-3','-4']:
            if First_step:
                sum_vector = np.array(d['doc_vector'][0][layer])
                First_step = False
            else :
                sum_vector += np.array(d['doc_vector'][0][layer])
        mean_vector = sum_vector/4
        Bert_vector.update({d['new_id']:mean_vector})

    Bert_data = pd.DataFrame(Bert_vector).T
    feature_size = Bert_data.shape[1]
    Bert_data.columns = ['BV_'+str(i) for i in range(0,feature_size)]

    return Bert_data

def generating_predict_data(df,log_data,user_feature_names,news_feaure_names,select_day):
    one_user = df[user_feature_names]

    time_diff = (one_user['time']- pd.to_datetime(log_data['create_date'])).dt.days
    condition_1 = (time_diff>=0) & (time_diff<=select_day)
    selected_news_1 = log_data[condition_1][news_feaure_names].drop_duplicates()
    target_news = pd.DataFrame(df[['news_id','create_date']]).T
    selected_news = pd.concat([target_news,selected_news_1])
        

    user_data = dict(one_user)
    news_data = dict(selected_news)

    predict_data = dict()
    predict_data.update(user_data)
    predict_data.update(news_data)
    result = pd.DataFrame(predict_data)

    return result

# data = predict_data_for_oneuser;handle_mms_name = ['sub_time'];news_lbe = news_lbe;mms= mms
def relative_news_handing(data,handle_mms_name,news_lbe,mms = None):
    #####與news有關的#####
    # covert string to datetime
    df = data.copy()
    df['create_date'] = pd.to_datetime(df['create_date'])

    # 選取大於0的
    df['sub_time'] =df['time'].sub(df['create_date'] , axis=0).dt.days
    #log_data =log_data[log_data['sub_time']>=0] 

    if not mms :
        mms = MinMaxScaler(feature_range=(0,1))
        mms.fit(log_data[handle_mms_name])
    df[handle_mms_name] = mms.transform(df.copy()[handle_mms_name])

    df['news_id'] = news_lbe.fit_transform(df['news_id'])
    ##############################
    return df , mms

def loadData():
    pkl_file = '/home/jovyan/at081-group39/python_code/1_deepctr_models/model-data.pkl'
    pkl_all_mclass = '/home/jovyan/at081-group39/data_processed/news/all_mclass.p'
    if os.path.isfile(pkl_file) :
        data = pd.read_pickle(pkl_file)
    else:
        df_userlog_raw = pd.read_pickle(os.path.join(DATA_ROOT,'userlog/user2newsid_list.pkl'))
        #df_userlog_raw = pd.read_csv(os.path.join(DATA_ROOT,'userlog/user2unewsid_list.csv'))
        #print(df_userlog_raw.head())
        df_userlog_raw['create_date'] = pd.to_datetime(df_userlog_raw['create_date'])
        df_userlog_raw['time'] = pd.to_datetime(df_userlog_raw['time'])
        
        # Caculate Time (Visit - Create)
        df_userlog_raw['sub_time'] = df_userlog_raw['time'].sub(df_userlog_raw['create_date'] , axis=0).dt.days
        print('Invalid Data:',df_userlog_raw[df_userlog_raw['sub_time'] < 0].shape[0],df_userlog_raw[df_userlog_raw['sub_time'] < 0].shape[0]/df_userlog_raw.shape[0])
        df_userlog_raw.describe()
        
        # Remove invalid Time (Visit - Create)
        data = df_userlog_raw[df_userlog_raw['sub_time']>=0] 
        
        # Train Split Data
        date_cut=pd.to_datetime("20181001 00:00:00")
        def split_train_valid(row):
            return True if pd.to_datetime(row) < date_cut else False

        data["is_train"] = data["time"].apply(split_train_valid)
        
        data.to_pickle(pkl_file)
      
    all_mclass = {}  
    with open(pkl_all_mclass, 'rb') as f:
        all_mclass = pickle.load(f)
            
    return data, all_mclass

# data = train_set
def generate_negative_sammple(data,user_feature_names,news_feaure_names):
    df = data.copy()
    df = df[user_feature_names+news_feaure_names]
    df['time'] = pd.to_datetime(df['time'])
    df['create_date'] = pd.to_datetime(df['create_date'])
    df['label'] = 1

    user_id_list = df['user_id'].unique().tolist()


    for num,user_id in enumerate(user_id_list):
        # user_id = 278
        temp_df = df.query('user_id == @user_id')

        view_min_date = temp_df['time'].min()
        view_max_date =temp_df['time'].max()
        last_date = temp_df['time'].max()

        # for one day user 
        if view_min_date.date() == view_max_date.date():
            view_min_date = view_min_date.date()
            view_max_date = view_max_date.date() + datetime.timedelta(days = 1)
        
        condition_1 = (df['create_date']<=  view_max_date) & (df['create_date']>=  view_min_date)
        candidate_news = df[condition_1][news_feaure_names].drop_duplicates()
        condtion_2 = ~candidate_news['news_id'].isin(temp_df['news_id'])
        candidate_news = candidate_news[condtion_2]
        sample_size = temp_df.shape[0]
        if  candidate_news.shape[0] == 0:
            continue
        candidate_news = candidate_news.sample(n = int(sample_size/2), random_state=1,replace=True)

        user_last_row = temp_df.query('time == @last_date').iloc[0,:]
        user_data = dict(user_last_row[user_feature_names])
        news_data =  dict(candidate_news)
        negative_data = dict()
        negative_data.update(user_data)
        negative_data.update(news_data)
        negative_data_df = pd.DataFrame(negative_data)
        negative_data_df['label'] = 0

        df = pd.concat([df,negative_data_df])

        if num%1000 == 0:
            print(num)

    return df















def generating_predict_data(df,news_data,user_feature_names,news_feaure_names,select_day):

    one_user = df[user_feature_names]

    time_diff = (one_user['time']- pd.to_datetime(news_data['create_date'])).dt.days
    condition_1 = (time_diff>=0) & (time_diff<=select_day)
    selected_news = news_data[condition_1][news_feaure_names].reset_index(drop=True)
    selected_news = selected_news.rename(columns={'new_id':'news_id'})

    user_data = dict(one_user)
    news_data = dict(selected_news)

    predict_data = dict()
    predict_data.update(user_data)
    predict_data.update(news_data)

    result = pd.DataFrame(predict_data)

    return result 

