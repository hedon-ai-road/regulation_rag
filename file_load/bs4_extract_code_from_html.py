from bs4 import BeautifulSoup

# 读取 HTML 文件内容
html_txt = ''
with open("./file_load/fixtures/test.html", 'r') as f:
    for line in f.readlines():
        html_txt += line


# 解析 HTML
soup = BeautifulSoup(html_txt, 'lxml')

# 代码块 td class="code"
code_content = soup.find_all('td', class_="code")
for ele in code_content:
    print(ele.text)
    print("+"*100)
