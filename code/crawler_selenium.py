import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup


options = webdriver.ChromeOptions()
options.add_argument("--incognito")

driver = webdriver.Chrome(options=options)

def get_text(sid,page):
    driver.get(f"https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1={sid}#&date=%2000:00:00&page={page}")

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    contents = soup.select('#section_body > ul > li > dl > dt > a')
    array=[]
    for element in contents:
        t=element.text
        if t in ['','동영상기사']:
            continue
        array.append(element.text)
    return array



# 0: IT과학, 1: 경제, 2: 사회, 3: 생활문화, 4: 세계, 6: 정치
# 105, 101, 102, 103, 104, 100
list_news=[]
tag_num=4
for page in range(100):
    list_news=list_news+get_text(104,page)
art_df=pd.DataFrame([list_news,[tag_num*len(list_news)]], columns=['text','target'])
art_df.to_csv(f'data/augmentation_{tag_num}.csv')
