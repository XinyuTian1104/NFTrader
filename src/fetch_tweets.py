import snscrape.modules.twitter as sntwitter
from tqdm import tqdm

def get_keyword(keyword, start_date, end_date):
    # scrape a keyword with progress bar
    tweets = []
    for i, tweet in enumerate(tqdm(sntwitter.TwitterSearchScraper(f'{keyword} since:{start_date} until:{end_date}').get_items())):
        tweets.append(tweet)
    return tweets
    
def main():
    data = get_keyword('Ethereum', '2021-01-01', '2021-01-02')
    print(data)

if __name__ == '__main__':
    main()