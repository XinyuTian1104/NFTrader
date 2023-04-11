import requests
from bs4 import BeautifulSoup
from PIL import Image
import re

header = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.0.0'
}

data_path = '/Users/crinstaniev/Courses/STATS402/data'

def fetch_nft(slug):
    URL = f'https://api-bff.nftpricefloor.com/projects/{slug}/charts/all'
    try:
        result = requests.get(URL, headers=header).json()
        print(result)
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
    response = requests.get(URL, headers=header)
    text = response['textEn']
    text = re.sub('<[^<]+?>', ' ', text).strip()
    text = re.sub(' +', ' ', text)

def fetch_images(soup):
    image = soup.find_all('img')[1]
    src = image.get('src')
    image = Image.open(requests.get(src, stream=True).raw)
    return image

def main():
    pass

if __name__ == '__main__':
    main()
    