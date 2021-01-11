from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import os
import re,os,base64
import requests
import zipfile
import time
import threading
import multiprocessing
import random

use_proxy = False

def requset_write_file(file_name,url):
    proxys = [
        None,
        {
            'http':"http://yen:yen@ec2-34-226-122-90.compute-1.amazonaws.com:3128",
            'https':"http://yen:yen@ec2-34-226-122-90.compute-1.amazonaws.com:3128"
        }
        ,
        {
            'http': "http://yen:yen@ec2-107-20-86-83.compute-1.amazonaws.com:3128",
            'https':"http://yen:yen@ec2-107-20-86-83.compute-1.amazonaws.com:3128"
        }       
    ]    
    
    x = random.randint(1,2)
    proxy = proxys[x]

    try:
        resp = requests.get(url,proxies=proxy ,timeout=10)
    except:
        print(x)
    try:
        html = requests.get(url)
        byte_data = html.content
        with open(file_name,'wb') as f:
            f.write(byte_data)
            f.flush()
            f.close()
    except:
        print(file_name,url)

def create_dir(category):
    folder_path = "./img/{category}".format(category=category)
    if (os.path.exists(folder_path) == False):
        os.makedirs(folder_path)
    else:
        pass

def get_chromedriver(use_proxy=False, user_agent=None, proxy_auth_plugin=None) :
    # you need to change driver_path by yourself
    driver_path = '/root/wei/ParallelProgramming/Final_Project/crawler/driver/chromedriver'
    
    chrome_options = webdriver.ChromeOptions()
    
    chrome_options.add_argument('--headless')
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("--disable-notifications")
    
    if use_proxy and proxy_auth_plugin:
        pluginfile = 'proxy_auth_plugin.zip'
        with zipfile.ZipFile(pluginfile, 'w') as zp:
            zp.writestr("manifest.json", manifest_json)
            print(manifest_json)
            zp.writestr("background.js", background_js)
        chrome_options.add_extension(pluginfile)

    if user_agent:
        chrome_options.add_argument('--user-agent=%s' % user_agent)

    
    #driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver",options=chrome_options)
    #driver.get_screenshot_as_file("save.png")
    driver = webdriver.Chrome(driver_path,options=chrome_options)
    
   
    return driver

def crawler_google_img_no_create_threads(keyword,user_proxy = None, proxy_auth_plugin = None):
    
    folder_path = "./img/{keyword}".format(keyword=keyword)

    url = "https://www.google.com.tw/search?q={keyword}&rlz=1C1CAFB_enTW617TW621&source=lnms&tbm=isch&sa=X&ved=0ahUKEwienc6V1oLcAhVN-WEKHdD_B3EQ_AUICigB&biw=1128&bih=863".format(keyword=keyword.replace('_','+'))
    driver = get_chromedriver(use_proxy=user_proxy,proxy_auth_plugin = proxy_auth_plugin)
    
    driver.get(url)
    driver.execute_script("window.scrollBy(0,1000)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,1000)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,1000)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,1000)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,1000)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,800)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,800)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,800)")
    time.sleep(0.5)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    imgs = soup.find_all('img')

    get_img_count = 0

    for index ,img in enumerate(imgs):
        need_request = False
        content = img.get('src')
        try:
            img_height = int(img.get('height'))
        except:
            continue
           
        if not content or img_height <= 50:
            continue
        if content.find("http") != -1:
            need_request = True
            img_type = ".jpg"
        elif content.find("jpeg") != -1:
            img_type = ".jpg"
        elif content.find("png") != -1:
            img_type = ".png"
        else:
            img_type = "other"
           
        
        file_name = folder_path + "/" + keyword + str(get_img_count) + img_type
        
        proxys = [
        None,
        {
            'http':"http://yen:yen@ec2-34-226-122-90.compute-1.amazonaws.com:3128",
            'https':"http://yen:yen@ec2-34-226-122-90.compute-1.amazonaws.com:3128"
        }
        ,
        {
            'http': "http://yen:yen@ec2-107-20-86-83.compute-1.amazonaws.com:3128",
            'https':"http://yen:yen@ec2-107-20-86-83.compute-1.amazonaws.com:3128"
        }       
        ]
        with open(file_name,'wb') as f:
            if need_request:
                if user_proxy:
                    x = random.randint(0,2)
                    proxy = proxys[x]
                else:
                    proxy = None

                html = requests.get(content,proxies=proxy)
                byte_data = html.content
            else:
                base64_data = re.sub('^data:image/.+;base64,', '', content)
                byte_data = base64.b64decode(base64_data)
            
            f.write(byte_data)
            f.flush()
            f.close()

        get_img_count+=1
        if get_img_count == 100:
            break

def crawler_google_img(keyword,user_proxy = None, proxy_auth_plugin = None):
    create_dir(keyword)
    folder_path = "./img/{keyword}".format(keyword=keyword)
    
    url = "https://www.google.com.tw/search?q={keyword}&rlz=1C1CAFB_enTW617TW621&source=lnms&tbm=isch&sa=X&ved=0ahUKEwienc6V1oLcAhVN-WEKHdD_B3EQ_AUICigB&biw=1128&bih=863".format(keyword=keyword.replace('_','+'))
    driver = get_chromedriver(use_proxy=user_proxy,proxy_auth_plugin = proxy_auth_plugin)
    
    driver.get(url)
    driver.execute_script("window.scrollBy(0,1000)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,1000)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,1000)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,1000)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,1000)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,800)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,800)")
    time.sleep(0.3)
    driver.execute_script("window.scrollBy(0,800)")
    time.sleep(0.5)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    imgs = soup.find_all('img')

    get_img_count = 0
    threads = []

    for index ,img in enumerate(imgs):
        need_request = False
        content = img.get('src')
        try:
            img_height = int(img.get('height'))
        except:
            continue
           
        if not content or img_height <= 50:
            continue
        if content.find("http") != -1:
            need_request = True
            img_type = ".jpg"
        elif content.find("jpeg") != -1:
            img_type = ".jpg"
        elif content.find("png") != -1:
            img_type = ".png"
        else:
            img_type = "other"
           
        
        file_name = folder_path + "/" + keyword + str(get_img_count) + img_type
       
        if need_request:
            threads.append(threading.Thread(target = requset_write_file, args = (file_name,content)))
            threads[-1].start()
        else:
            with open(file_name,'wb') as f: 
                base64_data = re.sub('^data:image/.+;base64,', '', content)
                byte_data = base64.b64decode(base64_data)
                f.write(byte_data)
                f.flush()
                f.close()

        get_img_count+=1

        if get_img_count == 100:
            break

    for i in range(len(threads)):
        threads[i].join()

def sequence_crawler():
    search_category = ['apple pie','chocolate_cake', 'donuts', 'hamburger' , 'hot_dog', 'ice_cream', 'pizza']
    total_start = time.time()

    for index , category in enumerate(search_category):
        start = time.time()
        crawler_google_img_no_create_threads(category)
        end = time.time()
        print(category + " 執行時間：%f 秒" % (end - start))

    total_end = time.time() 
    print("總執行時間：%f 秒" % (total_end - total_start))
    
def task(category,use_proxy):
    start = time.time()
    crawler_google_img(category)
    end = time.time()
    print(category + " 執行時間：%f 秒" % (end - start))

def multi_process_crawler(use_proxy = None):
    search_category = ['apple_pie','chocolate_cake', 'donuts', 'hamburger' , 'hot_dog', 'ice_cream', 'pizza']
    processes = []
    start = time.time()
    for i in range(len(search_category)):
        processes.append(multiprocessing.Process(target = task, args = (search_category[i],use_proxy)))
        processes[i].start()
    for i in range(len(search_category)):
        processes[i].join()
    end = time.time()
    print("執行時間：%f 秒" % (end - start))
    print("The number of CPU is:" + str(multiprocessing.cpu_count()))

def multi_thread_crawler(search_category, use_proxy = None):
    threads = []
    start = time.time()
    for i in range(len(search_category)):
        threads.append(threading.Thread(target = task, args = (search_category[i],use_proxy)))
        threads[i].start()
    for i in range(len(search_category)):
        threads[i].join()

    end = time.time()
    print("執行時間：%f 秒" % (end - start))
    
if __name__ == "__main__":
    multi_thread_crawler(search_category = ['apple_pie','chocolate_cake', 'donuts', 'hamburger' , 'hot_dog', 'ice_cream', 'pizza'])

    