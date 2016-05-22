import xml.dom.minidom
import os
import json
from os import listdir

class Tweet:
    
    def __init__(self, sid, uid, content, polarity, jsonData):
        self.sid= sid
        self.uid = uid
        self.content = content
        self.polarity = polarity
        self.jsonData = jsonData
        pass
        
class TweetParser:
    
    def __init__(self):
        pass
        
    def parse(self,inputFile, metadataPath):
        self.cache={}
        self.loadMetadataCache(metadataPath)
        dom = xml.dom.minidom.parse(inputFile)
        return self.handleTweets(dom.getElementsByTagName("tweets")[0])
    
    def loadMetadataCache(self, metadataPath):
        if os.path.exists(metadataPath):
            for user in listdir(metadataPath):
                jsonUidMap = {}
                for tweet in listdir(metadataPath + os.sep + user):
                    jsonUidMap[tweet]=self.loadJson(metadataPath + os.sep + user + os.sep + tweet)
                self.cache[user]=jsonUidMap
                
    def loadJson(self, json_path):
        data = {}
        with open(json_path) as data_file:
            try:    
                data = json.load(data_file)
            except ValueError:
                pass
                #print "Error al cargar fichero " +json_path    
        return data
                      
    def handleTweets(self, tweets_dom):
        tweets = []
        tweets_elements = tweets_dom.getElementsByTagName("tweet")
        for tweet_element in tweets_elements:
            tweets.append(self.handleTweet(tweet_element))
        return tweets
    
    def handleTweet(self, tweetnode):
        sentiments = tweetnode.getElementsByTagName("sentiments");
        polarity = []
        for sentiment in sentiments:
            polarity = self.handleSentiment(sentiment)
        sid = self.handleTweetId(tweetnode)
        uid = self.handleUserId(tweetnode)
        content = self.handleContent(tweetnode)
        jsonData = self.getJsonTweetFromCache(uid, sid) 
        tweet = Tweet(sid, uid, content, polarity, jsonData)
        return tweet

    def handleTweetId(self, tweetnode):
        tweetid_elements = tweetnode.getElementsByTagName("tweetid")
        tweetid = self.getText(tweetid_elements[0].childNodes)
        return tweetid
    
    def handleUserId(self, tweetnode):
        user_elements = tweetnode.getElementsByTagName("user")
        userId = self.getText(user_elements[0].childNodes)
        return userId    
        
    def handleContent(self, tweetnode):
        content_elements = tweetnode.getElementsByTagName("content")
        content = self.getText(content_elements[0].childNodes, True)
        return content     
        
    def handleSentiment(self, sentiment):
        polarity = []
        polarity_elements = sentiment.getElementsByTagName("polarity")
        for polarity_element in polarity_elements:
            polarity.append(self.handlePolarity(polarity_element))
        return polarity
        
    def handlePolarity(self, polarity): 
        value_elements = polarity.getElementsByTagName("value")
        value = self.getText(value_elements[0].childNodes)
        return value     
       
    def getText(self, nodelist, cdata=False):
        rc = []
        for node in nodelist:
            if node.nodeType == node.TEXT_NODE:
                rc.append(node.data.encode('utf-8'))
            if cdata:
                if node.nodeType == node.CDATA_SECTION_NODE:
                    rc.append(node.data.encode('utf-8'))
                    
        return ''.join(rc)    
        
    def getJsonTweetFromCache(self, uid, sid):
        jsonData = {}
        if self.cache:
            if uid in self.cache.keys():
                uidDict = self.cache[uid]
                if uidDict:
                    if sid in uidDict.keys():
                        jsonData = uidDict[sid]
        return jsonData
    