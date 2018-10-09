import json
import sys
import twitter

reload(sys)
sys.setdefaultencoding("utf-8")

tApi = twitter.Api(consumer_key='qZyBB4orj5gOZr0rQIrw40YPo',
                    consumer_secret='hmU52D9EXpJpb21TT5SU4rEcugEDT356Dak6VgxWIqoRrlGBBt',
                    access_token_key='1479876308-Ql1relkcw0VAhioYpty77kd46r46zS9rkG0mZGm',
                    access_token_secret='XsslUf0KBTCzYKXutMQ0hRrSZLKKZWldZkAcp6TlspbA6')


def inf(tweet):
    print '-------------------------'
    print 'Tweet ID: ', tweet['id']
    print 'Tweet Text: ', tweet['text']



def Tweets():
    query = '#Deadpool2 -filter:links'
    MAX_ID =993385763470376960

    for it in range(1):
        x=tApi.GetSearch(query,lang='en',count=5000, max_id=MAX_ID, result_type='recent')
        tweets = [json.loads(str(raw_tweet)) for raw_tweet in x]
        # print(tweets)
        for tweet in tweets:
            # print(tweet)
            if 'retweeted_status' not in tweet.keys():
                f = 'deadpool2.txt'
                with open(f, 'a+') as file:
                    x = tweet['text']
                    json.dump(x, file)
                    file.write('\n')
                    print inf(tweet)
            else:
                print ('-')


if __name__ == '__main__':
    print "\n\n\n!!!!!!!!!!!GETTING TWEETS!!!!!!!!!\n"
    Tweets()

