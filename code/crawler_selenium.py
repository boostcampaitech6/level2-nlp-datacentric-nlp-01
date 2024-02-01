import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
from datetime import date, timedelta, datetime



options = webdriver.ChromeOptions()
options.add_argument("--incognito")

driver = webdriver.Chrome(options=options)


def get_text(sid1,sid2,date):
    driver.get(f"https://news.naver.com/breakingnews/section/{sid1}/{sid2}?date={date}")

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    contents = soup.select('a > strong')
    array=[]
    for element in contents:
        t=element.text
        if t in ['','동영상기사']:
            continue
        array.append([element.text,date])
        if len(array)==2:
            break
    return array

news_tag={105:{731,226,227,230,732,283,229,228},101:{259,258,261,771,260,262,310,263},
          102:{249,250,251,254,252,'59b',255,256,276,257},103:{241,239,240,237,238,376,242,243,244,248,245},
          104:{231,232,233,234,322},100:{264,265,268,266,267,269}}

# 0: IT과학, 1: 경제, 2: 사회, 3: 생활문화, 4: 세계, 6: 정치
# 105, 101, 102, 103, 104, 100


sdate = date(2017,1,1)
edate = date(2023,12,31)

list_tag1=list(news_tag.keys())
list_tag_num=[0,1,2,3,4,6]
list_date=[(sdate+timedelta(days=x)).strftime('%Y%m%d') for x in range(0,(edate-sdate).days,3)]


n=0
list_news = []
tag_num,tag1=list_tag_num[n],list_tag1[n]
for tag2 in news_tag[tag1]:
    for date in list_date:
        try:
            list_news=list_news+get_text(tag1,tag2,date)
            list_news=list(set(list_news))
        except:
            pass

output_news=[x[0] for x in list_news]
output_date=[x[1] for x in list_news]
art_df=pd.DataFrame({'text':output_news,'date':output_date,'label':[tag_num]*len(output_news)})
art_df.to_csv(f'augments_{tag_num}.csv')

