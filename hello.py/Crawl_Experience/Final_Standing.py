import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import random
from bs4 import BeautifulSoup

# 使用重试机制设置
retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry)
session = requests.Session()
session.mount("https://", adapter)

# 设置浏览器选项（无头模式，不显示浏览器界面）
options = Options()
options.add_argument("--headless")  # 启用无头模式
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# 使用 Service 来指定 ChromeDriver 路径
service = Service(ChromeDriverManager().install())

# 启动浏览器并通过 webdriver-manager 安装 ChromeDriver
driver = webdriver.Chrome(service=service, options=options)

# 访问 Codeforces 网站，触发 Cloudflare 验证
driver.get("https://codeforces.com/")

# 等待 Cloudflare 验证通过
time.sleep(10)  # 等待 10 秒钟，以确保 Cloudflare 验证挑战通过

# 获取 Cookie 中的 cf_clearance
cookies = driver.get_cookies()
cf_clearance = None
for cookie in cookies:
    if cookie['name'] == 'cf_clearance':
        cf_clearance = cookie['value']
        break

# 打印出获取的 cf_clearance
print(f"cf_clearance: {cf_clearance}")

# 关闭浏览器
driver.quit()

# 更新请求头和 Cookie
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0",
    "X-Requested-With": "XMLHttpRequest",  # 必须的请求头
    "Origin": "https://codeforces.com",
    "Referer": "https://codeforces.com/contests?complete=true",  # 确保 Referer 是正确的
}

cookies = {
    'cf_clearance': cf_clearance,
}

# 更新会话的请求头和 cookies
session.headers.update(headers)
session.cookies.update(cookies)

# 请求 Codeforces 公开的比赛列表
url = "https://codeforces.com/contests?complete=true"
response = session.get(url, headers=headers)

if response.status_code == 200:
    print("Successfully fetched contests data!")

    # 解析 HTML 页面
    soup = BeautifulSoup(response.text, "html.parser")

    # 获取所有比赛信息
    contests = soup.find_all('div', class_='contest-row')

    print(f"Found {len(contests)} contests.")

    # 遍历每个比赛，获取 Final standings 页面 URL
    for contest in contests:
        contest_name = contest.find('a').text.strip()
        contest_id = contest['data-contestid']

        print(f"\nAccessing Final standings for {contest_name} (contest ID: {contest_id})")

        # 构建 Final standings 页面 URL
        standings_url = f"https://codeforces.com/contest/{contest_id}/standings"

        # 发送请求获取 Final standings 页面
        standings_response = session.get(standings_url, headers=headers)

        if standings_response.status_code == 200:
            # 解析 Final standings 页面
            standings_soup = BeautifulSoup(standings_response.text, "html.parser")

            # 查找所有题目的提交数和通过数
            problems = standings_soup.find_all('td', class_='smaller bottom dark')

            for problem in problems:
                # 提取提交数和通过数
                passed = problem.find('span', class_='cell-passed-system-test')
                total = problem.find('span', class_='notice')

                # 获取通过数和提交数
                passed_count = passed.text.strip() if passed else 'N/A'
                total_count = total.text.strip() if total else 'N/A'

                print(f"Problem: Passed = {passed_count}, Total = {total_count}")

            # 如果你想保存每场比赛的数据，可以将数据保存到文件
            with open(f"standings_{contest_id}.txt", 'w', encoding='utf-8') as f:
                f.write(f"Final Standings for {contest_name} (ID: {contest_id})\n")
                for problem in problems:
                    passed = problem.find('span', class_='cell-passed-system-test')
                    total = problem.find('span', class_='notice')

                    passed_count = passed.text.strip() if passed else 'N/A'
                    total_count = total.text.strip() if total else 'N/A'

                    f.write(f"Problem: Passed = {passed_count}, Total = {total_count}\n")

            print(f"Saved standings data for {contest_name} to file.")
        else:
            print(f"Failed to access standings for {contest_name}. Status Code: {standings_response.status_code}")

        # 随机等待 1 到 3 秒，模拟人类行为
        time.sleep(random.randint(1, 3))

    print("Finished processing all past contests.")

else:
    print(f"Failed to fetch contests data. Status Code: {response.status_code}")
