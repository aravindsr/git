import tweepy #https://github.com/tweepy/tweepy


#Twitter API credentials
consumer_key = "1v9IvWBjS4jw63Q7LeWQODNEb"
consumer_secret = "DJrt3ewjZbcAymRYQFo9BI3LnrirURtVUKLSker6DwSy0cjwAC"
access_key = "29897913-bZMbgrVZYJfxiSOCvEzuvmJnyjce0kKuwMrcH7x82"
access_secret = "w0TLUH7OvusVaSSI0Zo2f1uVexVr2WQZv3oIjOgE48okB"



	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	#authorize twitter, initialize tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)
	
	#initialize a list to hold all the tweepy Tweets
alltweets = []	
	
	#make initial request for most recent tweets (200 is the maximum allowed count)
new_tweets = api.user_timeline(screen_name = "harvard",count=200)
	
	#save most recent tweets
alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
while len(new_tweets) > 0:
	print("getting tweets before %s" % (oldest))
		
		#all subsiquent requests use the max_id param to prevent duplicates
	new_tweets = api.user_timeline(screen_name = "harvard",count=200,max_id=oldest)
		
		#save most recent tweets
	alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1
		
	print("...%s tweets downloaded so far" % (len(alltweets)))

f = open('tweetsplz3.txt','a',encoding="utf-8")
f.write('tweet_id'+'\t'+'creation_date'+'\t'+'tweet_text')
for tweet in alltweets:
    print(tweet)
    f.write('\n' + tweet.id_str+'\t'+tweet.created_at.strftime('%B %d, %Y, %r')+'\t'+tweet.text)
    f.flush()

f.close()

	
	#transform the tweepy tweets into a 2D array that will populate the csv	
outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]

