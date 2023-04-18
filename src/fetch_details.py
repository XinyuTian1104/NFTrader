import json
import os
import re
import time

import dotenv
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm

header = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.0.0'
}

dotenv.load_dotenv('.env')

data_path = os.getenv('DATA_PATH')

def fetch_nft(slug):
    URL = f'https://api-bff.nftpricefloor.com/projects/{slug}/charts/all'
    try:
        result = requests.get(URL, headers=header).json()
        return result
    except:
        raise Exception(f'Failed to fetch {slug}')
    
def fetch_soup(slug):
    URL = f'https://nftpricefloor.com/{slug}'
    response = requests.get(URL, headers=header)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup
    
def fetch_description(slug):
    URL = f'https://api-bff.nftpricefloor.com/projects/{slug}/details'
    response = requests.get(URL, headers=header).json()
    text = response['textEn']
    text = re.sub('<[^<]+?>', ' ', text).strip()
    text = re.sub(' +', ' ', text)
    return text

def fetch_images(soup):
    image = soup.find_all('img')[1]
    src = 'https://nftpricefloor.com' + str(image.get('src'))
    image = Image.open(requests.get(src, stream=True).raw)
    return image

def fetch_details():
    top_projects_df = pd.read_csv(f'{data_path}/processed/top_projects.csv')
    slugs = top_projects_df['slug'].tolist()
    for slug in tqdm(slugs):
        fail_counter = 0
        while fail_counter <= 5:
            try:
                nft_json = fetch_nft(slug)
                # create datapath/raw/slug folder if not exist
                if not os.path.exists(f'{data_path}/raw/{slug}'):
                    os.makedirs(f'{data_path}/raw/{slug}')
                # save nft json
                with open(f'{data_path}/raw/{slug}/time_series.json', 'w') as f:
                    json.dump(nft_json, f)
                # fetch image
                soup = fetch_soup(slug)
                image = fetch_images(soup)
                # save image
                image.save(f'{data_path}/raw/{slug}/image.png')
                description = fetch_description(slug)
                # save description to txt
                with open(f'{data_path}/raw/{slug}/description.txt', 'w') as f:
                    f.write(description)
                time.sleep(1)
                fail_counter = 6
                continue
            except:
                print('Failed to fetch', slug)
                fail_counter += 1
        
def main():
    fetch_details()

if __name__ == '__main__':
    main()
    