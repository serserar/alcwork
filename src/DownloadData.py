import sys
import os
import time
import datetime
import json
from twitter import *
from TweetParser import TweetParser, Tweet




def parse_train(inputh_path, metaDataPath):
    parser = TweetParser() 
    return parser.parse(inputh_path, metaDataPath)
    
def download_tweets(inputh_path, dest_path):    
    tweets = parse_train(inputh_path, dest_path)
    retry = True
    tweetId = "154100507349233665"
    print("Descargando datos !!!")
    CONSUMER_KEY='JEdRRoDsfwzCtupkir4ivQ'
    CONSUMER_SECRET='PAbSSmzQxbcnkYYH2vQpKVSq2yPARfKm0Yl6DrLc'

    MY_TWITTER_CREDS = os.path.expanduser('~/.my_app_credentials')
    if not os.path.exists(MY_TWITTER_CREDS):
        oauth_dance("Semeval sentiment analysis", CONSUMER_KEY, CONSUMER_SECRET, MY_TWITTER_CREDS)
        
    oauth_token, oauth_secret = read_token_file(MY_TWITTER_CREDS)
    t = Twitter(auth=OAuth(oauth_token, oauth_secret, CONSUMER_KEY, CONSUMER_SECRET))
    
    for tweet in tweets:
        if retry:
            if tweet.sid == tweetId:
                retry = False
                continue
            else:    
                continue
        try:
            response = t.statuses.show(_id=tweet.sid)
            json_text = json.dumps(response)  
            save_tweet(dest_path, tweet.uid, tweet.sid, json_text)
        except TwitterError as e:
            if e.e.code == 429:
                rate = t.application.rate_limit_status()
                reset = rate['resources']['statuses']['/statuses/show/:id']['reset']
                now = datetime.datetime.today()
                future = datetime.datetime.fromtimestamp(reset)
                seconds = (future-now).seconds+1
                if seconds < 10000:
                    sys.stderr.write("Rate limit exceeded, sleeping for %s seconds until %s\n" % (seconds, future))
                    time.sleep(seconds)
            else:
                save_tweet(dest_path, tweet.uid, tweet.sid, 'Not Available')


def save_tweet(destpath, uid, sid, tweet_json):
        directory = destpath + os.sep + uid
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_ = open(directory + os.sep + sid, 'w')
        file_.write(tweet_json)
        file_.close()

def main():
    #download_tweets( "../data/general-tweets-train-tagged.xml", "../data/train")
    download_tweets( "../data/general-tweets-test.xml", "../data/test")
        
if __name__ == '__main__':
    main()