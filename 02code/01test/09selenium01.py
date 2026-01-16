from selenium import webdriver
from selenium.webdriver.common.by  import By
import os
from dotenv import load_dotenv
import time

load_dotenv()
driver = webdriver.Chrome()
driver.get('https://example.com')

title = driver.find_element(By.TAG_NAME, 'h1').text
ptag = driver.find_element(By.TAG_NAME, 'p').text
print(f'title : {title}')
print(f'ptag : {ptag}')

secrect_key = os.getenv('SECRET_KEY')
print(secrect_key)

time.sleep(10)
driver.quit()
