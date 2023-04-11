import snscrape.modules.twitter as sntwitter

def get_keyword(keyword, start_date, end_date):
    # scrape a keyword
    tweets_list = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} since:{start_date} until:{end_date}').get_items()):
        if i > 10:
            break
        tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    return tweets_list

def main():
    data = get_keyword('Ethereum', '2021-01-01', '2021-01-02')
    print(data)

if __name__ == '__main__':
    main()