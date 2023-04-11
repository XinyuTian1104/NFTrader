import os

import dotenv
import pandas as pd

from utils.config import data_path

dotenv.load_dotenv('/Users/crinstaniev/Courses/STATS402/src/.env')
NUM_PROJECTS = int(os.getenv('NUM_PROJECTS'))


def process_projects():
    df = pd.read_json(f'{data_path}/raw/projects.json')
    df = df.sort_values(by='ranking').reset_index(drop=True)
    df = df[['name', 'slug', 'ranking']][:NUM_PROJECTS]
    df.to_csv(f'{data_path}/processed/top_projects.csv', index=False)

def main():
    process_projects()

if __name__ == '__main__':
    main()