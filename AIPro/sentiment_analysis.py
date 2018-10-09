import numpy as np
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score,average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import os

# Variables Initialization

Neg = []
Pos = []
vPos = []
vNeg = []
data_X = ""
data_Y = ""
test_pos=0
test_neg = 0
neut=0


"""
used to generate stopword list
it returns a Stopwords LIST
"""
def gSW():

    #get the Text File which has all the stopwords from the file
    get_stopWords = "AIPro/Data/stopwords.txt"

    #list for stopwords
    stopWords = []

    #open the file of stopwords open it and store it in a list

    o = open(get_stopWords, 'r')
    s_line = o.readline()
    while s_line:
        stopWords.append(s_line.strip())
        s_line = o.readline()
    o.close()


    return stopWords

"""
It is used to generate Lexicon of sentiment, from a text file
parameter path of the file to get lexicon from
returns affine_list
"""
def generateAffinityList(datasetLink):

    affin_dataset = datasetLink

    affin_list = open(affin_dataset).readlines()


    return affin_list

"""
    This function is used to create a Dictionary of words according to the polarities
    Every word from the AFFIN-111 Lexicon is categorized
    We have taken 4 Categories:
    Very Positive Words, Positive Words, Negative Words, Very Negative Words
    :parameter affin_list
"""
def createDictionaryFromPolarity(affin_list):

    # Create list to store the words and its score i.e. polarity
    words = []
    score = []

    #for word in affine_list, generate the Words with their scores (polarity)
    for word in affin_list:
        words.append(word.split("\t")[0].lower())
        score.append(int(word.split("\t")[1].split("\n")[0]))

    #categorize words into different catogories
    for elem in range(len(words)):
        if score[elem] == 1 or score[elem] == 2 or score[elem] == 3:
            Pos.append(words[elem])
        elif score[elem] == 4 or score[elem] == 5:
            vPos.append(words[elem])
        elif score[elem] == -1 or score[elem] == -2 or score[elem] == -3:
            Neg.append(words[elem])
        elif score[elem] == -4 or score[elem] == -5:
            vNeg.append(words[elem])


    # print(vNeg)
    # print(Neg)
    # print(vPos)
    # print(Pos)

"""
Here preprocessing of the data and dimensionality steps is done
:parameter Dataset
:returns processed_data :LIST
"""
def preprocessing(dataSet):

    processed_data = []

    #create a list of all the Stopwords to be removed
    stopWords = gSW()
    for tweet in dataSet:

        temp_tweet = tweet

        #Convert @username to USER_MENTION
        tweet = re.sub('@[^\s]+', 'USER_MENTION', tweet).lower()
        tweet.replace(temp_tweet , tweet)

        #Remove the unnecessary white spaces
        tweet = re.sub('[\s]+',' ', tweet)
        tweet.replace(temp_tweet, tweet)

        #Replace #HASTAG with only the word by removing the HASH (#) symbol
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

        #Replace all the numeric terms
        tweet = re.sub('[0-9]+', "",tweet)
        tweet.replace(temp_tweet,tweet)

        #Remove all the STOP WORDS
        for sw in stopWords:
            if sw in tweet:
                tweet = re.sub(r'\b' + sw + r'\b'+" ","",tweet)

        tweet.replace(temp_tweet, tweet)

        #Replace all Punctuations
        tweet = re.sub('[^a-zA-z ]',"",tweet)
        tweet.replace(temp_tweet,tweet)

        #Remove additional white spaces
        tweet = re.sub('[\s]+',' ', tweet)
        tweet.replace(temp_tweet,tweet)

        #Save the Processed Tweet after data cleansing
        processed_data.append(tweet)

    return processed_data

"""
Tbis function is used to create feature vector and assign class label accordingly
:parameter Tweet Dataset: dataset, Class Label: type_class
:returns feature_vector
"""
def FeaturizeTrainingData(dataset, type_class):

    neutral_list = []
    i=0

    # split every word of the Tweet for each tweet
    data = [tweet.strip().split(" ") for tweet in dataset]

    # used to store feature of tweets
    feature_vector = []
    train_pos = 0
    train_neg = 0
    # for every sentence i.e. TWEET find the words and their category
    for sentence in data:
        # Category count for every Sentence or TWEET
        vNeg_count = 0
        Neg_count = 0
        Pos_count = 0
        vPos_count = 0


        # for every word in sentence, categorize
        # and increment the count by 1 if found
        for word in sentence:
            if word in vPos:
                vPos_count = vPos_count + 1
            elif word in Pos:
                Pos_count = Pos_count + 1
            elif word in vNeg:
                vNeg_count = vNeg_count + 1
            elif word in Neg:
                Neg_count = Neg_count + 1
        i+=1

        #Assign Class Label
        if vPos_count == vNeg_count == Pos_count == Neg_count:
            feature_vector.append([vPos_count, Pos_count, Neg_count, vNeg_count, "neutral"])
            neutral_list.append(i)
        else:
            if type_class == "positive":
                train_pos = train_pos + 1
            else:
                train_neg = train_neg + 1
            feature_vector.append([vPos_count, Pos_count, Neg_count, vNeg_count, type_class])

    #print(neutral_list)
    return feature_vector, train_pos, train_neg

"""
    This function is used to generate the Feature Vectors for the Test Data
    :parameter Tweet Dataset: dataset
    :returns feature_vector
"""
def FeatureizeTestData(dataset):
    global neut,test_pos,test_neg
    data = [tweet.strip().split(" ") for tweet in dataset]

    feature_vector = []


    for sentence in data:
        vNeg_count = 0
        Neg_count = 0
        Pos_count = 0
        vPos_count = 0


        # for every word in sentence, categorize
        # and increment the count by 1 if found
        for word in sentence:
            if word in Pos:
                Pos_count = Pos_count + 1
            elif word in vPos:
                vPos_count = vPos_count + 1
            elif word in Neg:
                Neg_count = Neg_count + 1
            elif word in vNeg:
                vNeg_count = vNeg_count + 1

        if (vPos_count + Pos_count) < (vNeg_count + Neg_count):
            feature_vector.append([vPos_count, Pos_count, Neg_count, vNeg_count, "negative"])
            test_neg = test_neg+1

        elif (vPos_count + Pos_count) > (vNeg_count + Neg_count):
            feature_vector.append([vPos_count, Pos_count, Neg_count, vNeg_count, "positive"])
            test_pos = test_pos + 1

        else:
            feature_vector.append([vPos_count, Pos_count, Neg_count, vNeg_count, "neutral"])
            neut = neut+1
    return (feature_vector, test_pos, test_neg, neut)

"""
This function is used to classify data using Naive Bayes
"""
def classify_naive_bayes(train_X, train_Y, test_X):

    print("Classifying using Gaussian Naive Bayes ...")
    return GaussianNB().fit(train_X,train_Y).predict(test_X)


"""
It is used to classify data using SVM
"""
def classify_svm(train_X, train_Y, test_X):

    print("Classifying using Support Vector Machine ...")

    clf = SVC()
    clf.fit(train_X,train_Y)

    return clf.predict(test_X)

def classify_naive_bayes_twitter(train_X, train_Y, test_X, test_Y):

    print("Classifying using Gaussian Naive Bayes ...")
    gnb = GaussianNB()
    yHat = gnb.fit(train_X,train_Y).predict(test_X)

    conf_mat = confusion_matrix(test_Y,yHat)
    print(conf_mat)
    Accuracy = float(sum(conf_mat.diagonal())) / np.sum(conf_mat)
    print("Accuray: ", Accuracy)
    evaluate_classifier(conf_mat, Accuracy)



def classify_svm_twitter(train_X, train_Y, test_X, test_Y):

   print("Classifying using Support Vector Machine ...")
   clf = SVC()
   clf.fit(train_X, train_Y)
   yHat = clf.predict(test_X)
   print(confusion_matrix(test_Y, yHat))
   Accuracy = float(sum(confusion_matrix(test_Y, yHat).diagonal())) / np.sum(confusion_matrix(test_Y, yHat))
   print("Accuracy: ", Accuracy)
   evaluate_classifier(confusion_matrix(test_Y, yHat), Accuracy)



"""
 This function is used to classify tweets based on algorithm to classify
"""
def classify_twitter_data(file_name):

    test_data = open(dirPath+"/AIPro/Data/"+file_name).readlines()
    test_data = preprocessing(test_data)
    test_data,test_pos,test_neg,neu = FeatureizeTestData(test_data)
    test_data = np.reshape(np.asarray(test_data),newshape=(len(test_data),5))
    print("Positive tweets:",test_pos)
    print("Negative tweets:", test_neg)
    print(neu)

    #Split the Data into features and classes
    data_X_test = test_data[:,:4].astype(int)
    data_Y_test = test_data[:,4]

    print("Classifying", file_name)

    classify_svm_twitter(data_X, data_Y, data_X_test, data_Y_test)
    classify_naive_bayes_twitter(data_X, data_Y, data_X, data_Y)

    objects1 = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    y_pos1 = np.arange(len(objects1))
    per1 = [test_pos, test_neg, neu]
    plt.bar(y_pos1, per1, align='center', alpha=0.5)
    plt.xticks(y_pos1, objects1)
    plt.ylabel('COUNT')
    plt.show()



"""
It is used to evaluate the classifier's performance. 
Also calculate Precision, Recall, F-measure and Accuracy
"""
def evaluate_classifier(conf_mat,Accuracy):
    Precision = conf_mat[0,0]/float(sum(conf_mat[0]))
    Recall = conf_mat[0,0] / float(sum(conf_mat[:,0]))
    F_Measure = (2 * (Precision * Recall))/ (Precision + Recall)

    print("Precision: ",Precision)
    print("Recall: ", Recall)
    print("F-Measure: ", F_Measure)
    objects = ["ACCURACY", "PRECISION", "RECALL", "F-MEASURE"]
    y_pos=np.arange(len(objects))
    per=[Accuracy*100,Precision*100,Recall*100,F_Measure*100]
    plt.bar(y_pos,per,align='center',alpha=0.5)
    plt.xticks(y_pos,objects)
    plt.ylabel('VALUE')
    plt.show()




if __name__ == "__main__":
    #get the current directory
    os.chdir('../')
    dirPath = os.getcwd()
    #print(dirPath)

    # STEP 1: here affinity list is generated
    print("Please wait while we Classify your data ...")
    affin_list = generateAffinityList(dirPath+"/AIPro/Data/Affin_Data.txt")

    # STEP 2: here dictionary is created from lexicons
    createDictionaryFromPolarity(affin_list)

    # STEP 3: read positive and negative tweets and preprocessing is done
    print("Reading your data ...")
    positive_data = open(dirPath+"/AIPro/Data/rt-polarity-pos.txt").readlines()
    print("Preprocessing in progress ...")
    positive_data = preprocessing(positive_data)
    #print(positive_data)

    negative_data = open(dirPath+"/AIPro/Data/rt-polarity-neg.txt").readlines()
    negative_data = preprocessing(negative_data)
    #print(negative_data)

    # STEP 4: feature vector is created and class label is assigned for training data
    print("Generating the Feature Vectors ...")
    positive_sentiment, q, w = FeaturizeTrainingData(positive_data, "positive")
    print(len(positive_sentiment))
    negative_sentiment, e, r = FeaturizeTrainingData(negative_data, "negative")
    print(len(negative_sentiment))
    final_data = positive_sentiment + negative_sentiment
    print("Positive tweets tr:", q+e)
    print("Negative tweets tr:", w+r)
    final_data = np.reshape(np.asarray(final_data),newshape=(len(final_data),5))

    # #data is split into features and classes
    # data_X = final_data[:,:4].astype(int)
    # data_Y = final_data[:,4]



    #entire dataset is classified
    print("Training the Classifer according to the data provided ...")
    print("Classifying the Test Data ...")
    print("Evaluation Results will be displayed Shortly ...")
    file_name = "Twilight.txt"
    classify_twitter_data(file_name)
