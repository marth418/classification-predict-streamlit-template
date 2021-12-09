"""


    Simple Streamlit webserver application for serving developed classification
	models.


    Author: Explore Data Science Academy.


    Note:
    ---------------------------------------------------------------------

    Please follow the instructions provided within the README.md file

    located within this directory for guidance on how to use this script

    correctly.
    ---------------------------------------------------------------------


    Description: This file is used to launch a minimal streamlit web

	application. You are expected to extend the functionality of this script

	as part of your predict project.


	For further help with the Streamlit framework, see:


	https://docs.streamlit.io/en/latest/

"""

# Streamlit dependencies
import streamlit as st
import joblib, os
import pandas as pd
import numpy as np
import pickle 
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64


# Vectorizer
vectorizer = open("resources/CountVectoriser1.pkl","rb")
tweet_cv = joblib.load(vectorizer)  # loading your vectorizer from the pkl file

main_bg = "resources/twitter banner2.png"
main_bg_ext = "png"

side_bg = "resources/Twitter banner1.png"
side_bg_ext = "png"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("resources/output-onlinegiftools.gif", use_column_width=True)

st.markdown(
    """

<style>

.reportview-container .markdown-text-container {

    font-family: monospace;

}

.sidebar .sidebar-content {

    background-image: linear-gradient(#2e7bcf,#2e7bcf);

    color: transparent ;

}

.Widget>label {

    color: blue;

    font-family: monospace;

}

[class^="st-b"]  {

    color: black;

    font-family: monospace;
}

.st-bb {

    background-color: lightblue;

}

.st-at {

    background-color: ligtblue;

}

footer {

    font-family: monospace;

}

.reportview-container .main footer, .reportview-container .main footer a {

    color: white;

}

header .decoration {

    background-image: none;

}

</style>

""", unsafe_allow_html=True,)

# Load your raw data

raw = pd.read_csv("resources/train.csv")

train = raw.copy()

def pred_output(predict):
    if predict[0]==-1:
        output="Anti"
        st.error("Tweet Sentiment Categorized as: {}".format(output))
    elif predict[0]==0:
        output=" Neutral"
        st.info("Tweet Sentiment Categorized as: {}".format(output))
    elif predict[0]==1:
        output ="Pro"
        st.success("Tweet Sentiment Categorized as: {}".format(output))
    else:
        output = "News"
        st.warning("Tweet Sentiment Categorized as: {}".format(output))

# The main function where we will build the actual app

def main():

	"""Tweet Classifier App with Streamlit """

	# Creates a main title for page
	st.markdown("<h1 style='text-align: center; color: lightblue;'> Tweet Sentiment Classifier</h1>", unsafe_allow_html=True)
	st.write("<h3 style='text-align: center; color: lightblue ;'> Climate Change Classification</h3>", unsafe_allow_html=True)
	# Options for sidebar menu 
	options = ["Text Classification", "Model Insights", "Data Insights", 'About']
	selection = st.sidebar.selectbox("Choose Option", options)
    # Building out the predication page
	if selection == "Text Classification":
		models = ['Logistic Regression','Ridge Classifier','Stochastic Gradient Descent','Linear Support Vector']
		options1 = st.selectbox("Select Classification Model", models)

		if options1 =="Ridge Classifier":
			st.info("The Ridge Classifier,  based on Ridge regression method, converts the label data into [-1, 1] and solves the problem with regression method. The highest value in prediction is accepted as a target class and for multiclass data muilti-output regression is applied.")
			ridge_text = st.text_area("Enter Tweet for Prediction","Type Here")
			if st.button("Predict"):
				vect1_text = tweet_cv.transform([ridge_text]).toarray()
				pred = joblib.load(open(os.path.join("resources/RidgeClassifier.pkl"),"rb"))
				predict = pred.predict([ridge_text])
				pred_output(predict)
		if options1 =="Stochastic Gradient Descent":
			st.info("Stochastic Gradient Descent is a simple yet efficient optimization algorithm used to find the values of parameters/coefficients of functions that minimize a cost function. In other words, it is used for discriminative learning of linear classifiers under convex loss functions such as SVM and Logistic regression. It has been successfully applied to large-scale datasets because the update to the coefficients is performed for each training instance, rather than at the end of instances.")
			sgd_text = st.text_area("Enter Tweet for Prediction","Type Here")
			if st.button("Predict"):
				vect2_text = tweet_cv.transform([sgd_text]).toarray()
				pred = joblib.load(open(os.path.join("resources/SGDClassifier.pkl"),"rb"))
				predict = pred.predict([sgd_text])
				pred_output(predict)
		if options1 == "Linear Support Vector":
			st.info(" Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.")
			linear_text = st.text_area("Enter Tweet for Prediction","Type Here")
			if st.button("Predict"):
				vect3_text = tweet_cv.transform([linear_text]).toarray()
				pred = joblib.load(open(os.path.join("resources/LSVClassifier.pkl"),"rb"))
				predict = pred.predict([linear_text])
				pred_output(predict)
		elif options1 =="Logistic Regression":
			st.info("Logistic regression is a statistical analysis method used to predict a data value based on prior observations of a data set. Logistic regression has become an important tool in the discipline of machine learning. The approach allows an algorithm being used in a machine learning application to classify incoming data based on historical data. As more relevant data comes in, the algorithm should get better at predicting classifications within data sets")
			logistic_text = st.text_area("Enter Tweet for Prediction","Type Here")
			if st.button("Predict"):
				vect4_text = tweet_cv.transform([logistic_text]).toarray()
				pred = joblib.load(open(os.path.join("resources/LogisticRegression.pkl"),"rb"))
				predict = pred.predict([logistic_text])
				pred_output(predict)

 
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.	
	# Building out the "Information" page
	if selection == "About":
		st.info("This App provides a number of models that classify whether or not a person believes in climate change, based on their novel tweet data. The results provided on this App can be used by companies to access a broad base of consumer sentiment, spanning multiple demographic and geographic categories - thus increasing their insights and informing future marketing strategies.")
		# You can read a markdown file
		#from supporting resources folder
		st.image("Climate_Action_Poster.png", use_column_width=True)
		
		st.subheader("Raw Twitter data")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		st.subheader("Data Collection ")
		st.markdown("""<div> The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were collected. Each tweet is labelled as one of the following classes:</div>""",unsafe_allow_html=True)
		
		st.subheader('Class Description:')
 
		st.markdown("""
					<li>
						News(2): the tweet links to factual news about climate change
					</li>
				""",unsafe_allow_html=True)
		
		st.markdown("""
					<li>
						Pro(1): the tweet supports the belief of man-made climate change
					</li>
				""",unsafe_allow_html=True)
		st.markdown("""
					<li>
						Neutral(0): the tweet neither supports nor refutes the belief of man-made climate change
					</li>
				""",unsafe_allow_html=True)

		st.markdown("""
					<li>
						Anti(-1): the tweet does not believe in man-made climate change
					</li>
				""",unsafe_allow_html=True)

	def title_tag(title):
         html_temp = """<div style="background-color:{};padding:10px;border-radius:10px; margin-bottom:15px;"><h2 style="color:#lightblue;text-align:center;">"""+title+"""</h2></div>"""
         st.markdown(html_temp, unsafe_allow_html=True)
		
	if selection == "Data Insights":
		
		st.info("General Insights Gathered from Exloring the Dataset")
		insight = ['Senitment Distributions','Hashtags','Mentions']
		opt = st.selectbox("Select Insight", insight)

		if opt == 'Senitment Distributions':
			title_tag("Assessing Sentiment Distribution")
			fig = plt.figure(figsize=(8,6)) 
			colors = ['skyblue','skyblue','skyblue', 'skyblue']
			train.groupby('sentiment').message.count().sort_values().plot.barh(ylim=0, color=colors, title= 'NUMBER OF TWEETS IN EACH SENTIMENT CATEGORY\n')
			plt.xlabel('Number of ocurrences', fontsize = 10);
			st.pyplot(fig)
			
			st.markdown("""<div> The data is unevenly distributed, Pro sentiments are overly represented as comapred to the other sentiments. However, this could be just be an indication that there are more people who believe in climate change than those who don't and those who present neutral on the matter. </div>""",unsafe_allow_html=True)
			st.image("resources/pie-chart.png", use_column_width=True)

		if opt == 'Hashtags':

			title_tag("Showing Popular Hashtags based of Sentiment")
			####### Splitting the df into its sentiments
			non_believer = train[train.sentiment == -1]
			believer = train[train.sentiment == 1]
			news = train[train.sentiment == 2]
			neutral = train[train.sentiment == 0]
			
			####### Extracting words which start with a hash

			non_believers_hashtags = non_believer.message.str.extractall(r'(\#\w+)')[0].value_counts()
			believe_hashtags = believer.message.str.extractall(r'(\#\w+)')[0].value_counts()
			news_hashtags = news.message.str.extractall(r'(\#\w+)')[0].value_counts()
			neutral_tweets_hashtags = neutral.message.str.extractall(r'(\#\w+)')[0].value_counts()
			
			st.markdown("<h4 style='color:lightblue; text-align:center !important'> Believer Hashtags</h4>",unsafe_allow_html=True)
			wordcloud1 = WordCloud(width=800, height=500, random_state=21,colormap="Greens_r", max_font_size=110).generate_from_frequencies(believe_hashtags.to_dict())
			fig = plt.figure(figsize=(10, 7))
			plt.imshow(wordcloud1, interpolation="bilinear")
			plt.axis('off')
			st.pyplot(fig)
			
			st.markdown("<h4 style='color:lightblue; text-align:center !important'> Non-Believer Hashtags</h4>",unsafe_allow_html=True)
			wordcloud2 = WordCloud(width=800, height=500, random_state=21,colormap="OrRd_r", max_font_size=110).generate_from_frequencies(non_believers_hashtags.to_dict())
			fig = plt.figure(figsize=(10, 7))
			plt.imshow(wordcloud2, interpolation="bilinear")
			plt.axis('off')
			st.pyplot(fig)

			st.markdown("<h4 style='color:lightblue; text-align:center !important'> News Hashtags</h4>",unsafe_allow_html=True)
			wordcloud3 = WordCloud(width=800, height=500, random_state=21,colormap="cool_r", max_font_size=110).generate_from_frequencies(news_hashtags.to_dict())
			fig = plt.figure(figsize=(10, 7))
			plt.imshow(wordcloud3, interpolation="bilinear")
			plt.axis('off')
			st.pyplot(fig)
			
			st.markdown("<h4 style='color:lightblue; text-align:center !important'> Neutral Hashtags</h4>",unsafe_allow_html=True) 
			wordcloud4 = WordCloud(width=800, height=500, random_state=21,colormap="RdYlBu", max_font_size=110).generate_from_frequencies(neutral_tweets_hashtags.to_dict())
			fig = plt.figure(figsize=(10, 7))
			plt.imshow(wordcloud4, interpolation="bilinear")
			plt.axis('off')
			st.pyplot(fig)
			
			st.markdown("""<div> There are numerous hashtags that are frequently used in all four sentiment categories. For instance, climate and change are common hashtags regardless of the users sentiment. However, there are a couple of hashtags that seem to be uniquely linked to a specific sentiment category , for example, #MAGA being used by people who do not believe in climate change and #ImVotingBecause being linked to believers.</div>""",unsafe_allow_html=True)

		if opt == 'Mentions':
				
			title_tag("Showing Popular Mentions based of Sentiment")

			####### Splitting the df into its sentiments
			non_believer = train[train.sentiment == -1]
			believer = train[train.sentiment == 1]
			news = train[train.sentiment == 2]
			neutral = train[train.sentiment == 0]
				
			####### Extracting users
			non_believers_users = non_believer.message.str.extractall(r'(\@\w+)')[0].value_counts()
			believe_users= believer.message.str.extractall(r'(\@\w+)')[0].value_counts()
			news_users = news.message.str.extractall(r'(\@\w+)')[0].value_counts()
			neutral_users = neutral.message.str.extractall(r'(\@\w+)')[0].value_counts()
				
			st.markdown("<h4 style='color:lightblue; text-align:center !important'> Believer Mentions</h4>",unsafe_allow_html=True)
			Wordcloud1 = WordCloud(width=800, height=500, random_state=21,colormap="Greens_r", max_font_size=110).generate_from_frequencies(believe_users.to_dict())
			fig = plt.figure(figsize=(10, 7))
			plt.imshow(Wordcloud1, interpolation="bilinear")
			plt.axis('off')
			st.pyplot(fig)
				
				
			st.markdown("<h4 style='color:lightblue; text-align:center !important'> Non-Believer Mentions</h4>",unsafe_allow_html=True)
			Wordcloud2 = WordCloud(width=800, height=500, random_state=21,colormap="OrRd_r", max_font_size=110).generate_from_frequencies(non_believers_users.to_dict())
			fig = plt.figure(figsize=(10, 7))
			plt.imshow(Wordcloud2, interpolation="bilinear")
			plt.axis('off')
			st.pyplot(fig)
				
	
			st.markdown("<h4 style='color:lightblue; text-align:center !important'> News Mentions</h4>",unsafe_allow_html=True)
			Wordcloud3 = WordCloud(width=800, height=500, random_state=21,colormap="cool_r", max_font_size=110).generate_from_frequencies(news_users.to_dict())
			fig = plt.figure(figsize=(10, 7))
			plt.imshow(Wordcloud3, interpolation="bilinear")
			plt.axis('off')
			st.pyplot(fig)

			st.markdown("<h4 style='color:lightblue; text-align:center !important'> Neutral Mentions</h4>",unsafe_allow_html=True)
			Wordcloud4 = WordCloud(width=800, height=500, random_state=21,colormap="RdYlBu", max_font_size=110).generate_from_frequencies(neutral_users.to_dict())
			fig = plt.figure(figsize=(10, 7))
			plt.imshow(Wordcloud4, interpolation="bilinear")
			plt.axis('off')
			st.pyplot(fig)

			st.markdown("""<div> Mentions are also a very important part of gaining user insight. We can paint a pciture what a user believes by observing who they mention in thier tweets. The above word clouds display the popular mentions in each snetiment catogory. For instance, someone who mentions @Fox News is more likely to be a non-believer whilst someone who mentions @ Bernie Sanders is likely to be a believer.  </div>""",unsafe_allow_html=True)
			

	if selection == "Model Insights":
		st.info("Analysing model performaces ")
	
		title_tag("Linear Support Vector Evaluation")
		st.markdown("<h4 style='color:lightblue; text-align:center !important'>Classification Report</h4>",unsafe_allow_html=True)
		st.image('resources/Lsvc Classification Report.png',width = 500)
		st.markdown("<h4 style='color:lightblue; text-align:center !important'>Confusion Matrix</h4>",unsafe_allow_html=True)
		st.image('resources/Lsvc1 Confusion Matrix.png', width = 500)
		   
		title_tag("Logistic Regression Evaluation")
		st.markdown("<h4 style='color:lightblue; text-align:center !important'>Classification Report</h4>",unsafe_allow_html=True)
		st.image('resources/Logistic Classification Report.png',width = 500)
		st.markdown("<h4 style='color:lightblue; text-align:center !important'>Confusion Matrix</h4>",unsafe_allow_html=True)
		st.image('resources/logistic1 Confusion Matrix.png',width = 500)
		   
		title_tag("Stochastic Gradient Descent Evaluation")
		st.markdown("<h4 style='color:lightblue; text-align:center !important'>Classification Report</h4>",unsafe_allow_html=True)
		st.image('resources/SGD Classification Report.png',width = 500)
		st.markdown("<h4 style='color:lightblue; text-align:center !important'>Confusion Matrix</h4>",unsafe_allow_html=True)
		st.image('resources/SGD1 Confusion Matrix.png',width = 500)
		   
		title_tag("Ridge Classifier Evaluation")
		st.markdown("<h4 style='color:lightblue; text-align:center !important'>Classification Report</h4>",unsafe_allow_html=True)
		st.image('resources/Ridge Classification Report.png',width = 500)
		st.markdown("<h4 style='color:lightblue; text-align:center !important'>Confusion Matrix</h4>",unsafe_allow_html=True)
		st.image('resources/Ridge1 Confusion Matrix.png',width = 500)

		title_tag("Comparing Models")
		st.markdown("<h4 style='color:lightblue; text-align:center !important'>Dataframe Containing Model Performances</h4>",unsafe_allow_html=True)
		st.image('resources/model Performances.png',width = 500)

		st.markdown("""<div>
				<h5 style='color:lightblue'> Key Observations</h5>
				<ul>
					<li>
						The models predicts Pro and News tweets very well, on average the f1-score for pro and news tweets is 0.84 and 0.80 respectively. Anti and neutral tweets perform on the lover end averaging f1-scores of 0.60 and 0.55 repectively.
						
			
				""",unsafe_allow_html=True)
		st.markdown("""
					<li>
						The models have a combined average mean f1-score of 0.78, and combined overall average accuracy of 0.79.
					</li>
				""",unsafe_allow_html=True)
		
		st.markdown("""
					<li>
						On the training data the models performed more or less the same with notable but little difference as shown by above tables. However, on Unseen data the ridge and logistic classifiers outperform the linear and Stochastic Gradient Descent classifiers. 
					</li>""",unsafe_allow_html=True)
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':

	main()


