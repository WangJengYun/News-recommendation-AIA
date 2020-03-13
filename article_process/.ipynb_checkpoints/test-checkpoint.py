import os
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
os.chdir('/home/jovyan/at081-group39')
# ----------------------------------------
def reading_log_file(path):
    file_paths = [os.path.join(path,file_name) for file_name in os.listdir(path) if file_name.endswith('.csv')]    
    
    Reading_first_file = True
    for file_path in file_paths : 
        temp_df = pd.read_csv(file_path)
        if Reading_first_file == True :
            env_df = temp_df
            Reading_first_file = False
        else:
            env_df = pd.concat([env_df,temp_df])
        
    return env_df

def reading_industry_map(path):
# reading the industry_map 
    with open(path,'rb') as file :
        industry_map = pickle.load(file)
    return(industry_map)

env_df = reading_log_file('./sysjust03/NEWS_2nd/env')
newslog_df = reading_log_file('./sysjust03/NEWS_2nd/newslog')
print('env_df = {}'.format(env_df.shape))
print('newslog_df = {}'.format(newslog_df.shape))

industry_map = reading_industry_map('./sysjust03/NEWS_2nd/industry_map.p')
print(len(industry_map))
#===================================
#===================================
print('user_num :{}'.format(len(env_df['user_id'].unique())))
bot_crawler_agent = ['GOOGLEBOT','AHREFSBOT','NET/BOT','YANDEXBOT','APPLEBOT','YISOUSPIDE','WEBCRAWLER', 'APPCRAWLER']
not_webrawler = ~ env_df['config_browser_name'].isin(bot_crawler_agent)
#bot_user  = env_df['user_id'][~not_webrawler].unique()
#env_df_M1 = env_df[~env_df['user_id'].isin(bot_user)]
env_df_M1 = env_df[not_webrawler]
print('M1_user_num :{}'.format(len(env_df_M1['user_id'].unique())))

table = env_df[['user_id','config_browser_name']].drop_duplicates()
table.groupby('user_id').agg({'config_browser_name': lambda x: np.sum(x.isin(bot_crawler_agent))})

table.groupby('user_id').agg('config_browser_name':'size')
.apply(lambda x : x['config_browser_name'].isin(bot_crawler_agent))


bot_user  = env_df['user_id'][~not_webrawler].unique()
newslog_df.columns

newslog_df['time'] = pd.to_datetime(newslog_df['time'], format='%Y-%m-%d %H:%M:%S')
newslog_df['date'] = newslog_df['time'].dt.date
newslog_df['hours'] = newslog_df['time'].dt.hour
newslog_df['minute'] = newslog_df['time'].dt.minute

table_1 = newslog_df.groupby(['user_id','date']).apply(lambda x : pd.DataFrame({'reading_article':[len(x['guid'].unique())],'num_record':[x.shape[0]]}))

df = pd.DataFrame({"a":[1, 2, 1, 2], "b":[1, 2, 3, 4], "c":[5, 6, 7, 8]})

pd.DataFrame({'reading_article':len(df['b'].unique()),'num_record':df.shape[0]})

df.groupby(['a']).apply(lambda x : pd.DataFrame({'reading_article':len(x['b'].unique()),'num_record':x.shape[0]},index = 0))

all_table = all_table.groupby('user_id').apply(lambda x: x.sort_values(["count"], ascending = False))


table = newslog_df.groupby(['user_id','date']).size().sort_values(ascending=False)
table.name = 'count'
table = table.reset_index()
#table.query('user_id == "b\'79656e656b756f\'"')
temp_table = table.groupby('user_id').size().sort_values(ascending=False)
temp_table.name = 'day_num'
temp_table = temp_table.reset_index()
all_table = table.merge(temp_table,how = 'left',on = 'user_id')
all_table = all_table.groupby('user_id').apply(lambda x: x.sort_values(["count"], ascending = False))
conditional = all_table['count']>300
userID = all_table[conditional].user_id.unique()
AAA = all_table[all_table.user_id.isin(userID)]

AAA.to_csv('output.csv')
#======================
print('news_df_distinct = {}'.format(newslog_df.drop_duplicates().shape))
print('news_env_df = {}'.format(news_env_df.shape))


table = news_env_df.groupby(['user_id','time','guid']).size()
table = table.reset_index(name='counts')

TT = table.sort_values('counts',ascending=False)

statement = "user_id=='d890d0b0-08e4-4082-a111-5851ea5b3d62' & time == '2018-06-03 17:00:01' & guid == '10db5cc8-cac9-4655-a6ac-bf62af7710ae'"

newslog_df.query(statement)

news_env_df.query(statement)


newslog_df.drop_duplicates().shape


env_df['config_browser_name'].value_counts()
