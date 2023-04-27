import pandas as pd
import requests
import json

header = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.0.0'
}

data_path = '/Users/crinstaniev/Courses/STATS402/data'

def fetch_top_projects():
    URL = 'https://api-bff.nftpricefloor.com/projects'
    result = requests.get(URL, headers=header)
    if result.status_code == 200:
        result_json = result.json()
        return result_json
    else:
        raise Exception('Failed to fetch projects')
    
def main():
    projects = fetch_top_projects()
    with open(f'{data_path}/raw/projects.json', 'w') as f:
        json.dump(projects, f)

if __name__ == '__main__':
    main()