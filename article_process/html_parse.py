from lxml import etree
import os
import pandas as pd
import re
import pickle

def finding_all_newspath(news_path):
    news_date_folder = os.listdir(news_path)
    news_date_path = [os.path.join(news_path,i) for i in news_date_folder ]
    news_info  = pd.DataFrame(columns = ['date','new_id','path'])
    for folder_name,path in zip(news_date_folder,news_date_path) : 
        # path = news_date_path[123]
        files = list(filter(lambda x : x.endswith('.txt') ,os.listdir(path)))
        news_path = [os.path.join(path,i) for i in files]
        news_id = [re.findall('(.+)(?:\.txt)', i )[0] for i in files]
        temp_df = pd.DataFrame({'date':folder_name,'new_id':news_id,'path':news_path})
        news_info = pd.concat([news_info,temp_df],ignore_index=True)
    return news_info

def parsing_element(context):
    for  line in context : 
        if str(type(line)) == "<class 'lxml.etree._ElementUnicodeResult'>":
            yield line
        else :
            yield ''.join(line.xpath('.//text()'))


if __name__ == "__main__":
    os.chdir('/home/jovyan/at081-group39')
    news_table = finding_all_newspath('./sysjust03/NEWS_2nd/news')

    # setting
    sep = '<SEP>'
    del_re = "(^.+新聞.+記者 .+ (報導|.+報導))|(^＊編者按.+) |(^.+新聞.+ (報導|.+報導))"
    saving_folder = './sysjust03/NEWS_2nd/processed_news/news/'
    saving_all_news_path = './sysjust03/NEWS_2nd/processed_news/all_news.p'
    os.makedirs(saving_folder, exist_ok=True)

    news_num = news_table.shape[0]
    result = []
    for index in range(0, news_num):
        news_record = news_table.iloc[index]
        new_path = news_record['path']
#         new_path = news_table.query('new_id == "bf3f9f59-773f-42a7-896e-082675c8c6a7"')['path'].values[0]
#         reading text and data preparation
        with open(new_path, "r") as news:
            original_content = news.readlines()

        process_1 = [i.strip('\n').strip(' ').strip() for i in original_content]
        title = process_1.pop(0)
        process_2 = ''.join(process_1)

        # parsing html
        et = etree.HTML(process_2)
        lines_element = et.xpath('//p | //body/text() | //div |//font//text()')
        lines_generator = parsing_element(lines_element)
        lines_list = [line.strip() for line in lines_generator if not bool(re.search(del_re,line))]
        lines_list = list(filter(None, lines_list))
        combining_lines = sep.join(lines_list)

        # strong sentence
        strong_lines_list = et.xpath('//strong/text()')
        combining_strong_lines = sep.join(strong_lines_list)

        # saving the result
        news_dict = {'date':news_record['date'],
            'new_id':news_record['new_id'],
            'title':title,
            'content':combining_lines,
            'strong':combining_strong_lines}
        result.append(news_dict)
        
        # saving news into pickle 
        saving_file_name = "{}_{}.p".format(news_record['date'],news_record['new_id'])
        saving_news_path = saving_folder + saving_file_name
        with open(saving_news_path,'wb') as file:
            pickle.dump(news_dict,file) 

        print('-----{} : {} {}-----'.format(index,news_record['date'],title))
    
    # saving all news into  pickle 
    with open(saving_all_news_path,'wb') as file:
        pickle.dump(result,file)

with open(saving_all_news_path,'rb') as file:
        AA = pickle.load(file)
len(AA)        