#%%
import json
import multiprocessing
import os
import time

import pandas as pd
import requests
from tqdm import tqdm

#%%
header = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.0.0'
}

data_path = '/Users/crinstaniev/Courses/STATS402/data'


#%%
def fetch_top_projects():
    URL = 'https://api-bff.nftpricefloor.com/projects'
    result = requests.get(URL, headers=header)
    if result.status_code == 200:
        result_json = result.json()
        return result_json
    else:
        raise Exception('Failed to fetch projects')


# %%
top_projects = fetch_top_projects()
# %%
projects = []
for project in top_projects:
    name = project['name']
    ranking = project['ranking']
    slug = project['slug']
    projects.append(dict(
        ranking=ranking,
        slug=slug,
        name=name
    ))
# %%
top_projects_df = pd.DataFrame(projects).sort_values('ranking').reset_index(drop=True)
top_projects_df.to_csv(os.path.join(data_path, 'top_projects.csv'), index=False)
# %%

def fetch_nft(slug):
    URL = f'https://api-bff.nftpricefloor.com/projects/{slug}/charts/all'
    try:
        result = requests.get(URL, headers=header).json()
        data_processed = []
        for i in range(len(result['timestamps'])):
            data_processed.append(dict(
                timestamp=result['timestamps'][i],
                slug=result['slug'],
                floor_eth=result['floorEth'][i],
                floor_usd=result['floorUsd'][i],
                sales_count=result['salesCount'][i],
                volume_eth=result['volumeEth'][i],
                volume_usd=result['volumeUsd'][i],
            ))
        data_processed_df = pd.DataFrame(data_processed)
        return data_processed_df
    except:
        raise Exception(f'Failed to fetch {slug}')
# %%
top_project_slugs = top_projects_df[:100]['slug'].tolist()
# %%
top_nft_data = []
def _fetch_nft(slug):
    global top_nft_data
    try:
        nft_df = fetch_nft(slug)
        top_nft_data.append(nft_df)
        print('Fetched', slug)
    except:
        print('Failed to fetch', slug)
    return

if __name__ == '__main__':
    for slug in tqdm(top_project_slugs):
        _fetch_nft(slug)
        time.sleep(1)
        
    # concat and save
    top_nft_data_df = pd.concat(top_nft_data)
    top_nft_data_df.to_csv(os.path.join(data_path, 'top_nft_data.csv'), index=False)