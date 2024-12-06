import requests
import time
import random
from datetime import datetime, timezone

# 伪装客户端，模拟浏览器行为
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
}

# API URL，获取所有比赛列表
url = "https://codeforces.com/api/contest.list"

# 发送 GET 请求
response = requests.get(url, headers=headers)

# 检查请求是否成功
if response.status_code == 200:
    data = response.json()  # 将响应转换为 JSON 格式
    if data['status'] == 'OK':
        contests = data['result']  # 获取所有比赛列表
        print(f"Found {len(contests)} contests.")

        # 筛选出已结束的比赛（phase 为 'FINISHED'）
        past_contests = [contest for contest in contests if contest['phase'] == 'FINISHED']

        print(f"Found {len(past_contests)} past contests.")

        # 将比赛信息存储到 Contest.txt 文件
        with open('Contest.txt', 'w', encoding='utf-8') as f:
            for contest in past_contests:
                contest_name = contest['name']
                start_time = contest['startTimeSeconds']

                # 将 Unix 时间戳转换为带时区的日期时间格式
                start_time_str = datetime.fromtimestamp(start_time, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

                # 将比赛信息写入文件
                f.write(f"Contest: {contest_name}\n")
                f.write(f"Start Time: {start_time_str}\n\n")

        print("Past contest data has been saved to 'Contest.txt'.")
    else:
        print(f"Error: {data['comment']}")
else:
    print(f"Failed to fetch data. Status Code: {response.status_code}")

# 随机等待 1 到 3 秒，模拟人类行为
time.sleep(random.randint(1, 3))
