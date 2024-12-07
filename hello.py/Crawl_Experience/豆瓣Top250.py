import requests
import csv
import re

url = "https://movie.douban.com/top250"
head = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0"
}
response = requests.get(url, headers=head)
page_content = response.content.decode(encoding="utf-8")

# 解析数据
obj = re.compile(r'<li>.*?<div class="item">.*?<span class="title">(?P<title>.*?)'
                 r'</span>.*?<p class="">.*?<br>(?P<year>.*?)&nbsp.*?<span'
                 r'class="rating_num" property="v:average">(?P<score>.*?)</span>.*?'
                 r'<span>(?P<comment>.*?)人评价</span>', re.S)

# 开始匹配
result = obj.finditer(page_content)
f = open("Douban_Top250_Movies.csv", mode="w")
f_csv = csv.writer(f)  # 将文件以csv格式写入
for it in result:
    # print(it.group("title"))
    # print(it.group("score"))
    # print(it.group("comment"))
    # print(it.group("year").strip())
    dic = it.groupdict()  # 将迭代器中的内容以字典形式写入
    dic['year'] = dic['year'].strip()  # 去除年份前空格重新写入
    f_csv.writerow(dic.values())
f.close()
print("Successfully!")

