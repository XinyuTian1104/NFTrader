import os
import warnings

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from utils.config import data_path

warnings.filterwarnings('ignore')


def get_desc(slug, driver):
    url = f'https://nftpricefloor.com/{slug}'
    driver.get(url)
    paras = driver.find_elements_by_xpath("//p[@class='mt-4 dark:text-white']")
    return paras[0].text


def fetch_desc():
    options = Options()
    options.headless = True

    chrome_options = webdriver.ChromeOptions()
    # this will disable image loading
    chrome_options.add_argument('--blink-settings=imagesEnabled=false')

    driver = webdriver.Chrome(options=options, chrome_options=chrome_options)
    raw_data_dir = os.path.join(data_path, 'raw')
    dirs = os.listdir(raw_data_dir)
    # remove projects.json
    dirs.remove('projects.json')
    for slug in dirs:
        # check if there is description.txt
        desc_path = os.path.join(raw_data_dir, slug, 'description.txt')
        if not os.path.exists(desc_path):
            print(f'fetching description for {slug}')
            desc = get_desc(slug, driver)
            print(f'fetched description: {desc}')
            # save to description.txt
            with open(desc_path, 'w') as f:
                f.write(desc)
    print('all done')
    driver.close()


if __name__ == '__main__':
    fetch_desc()
