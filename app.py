# import streamlit as st

# def main():
#     st.write('dfvbasdvasdNLP APP With Streamlit')
#     menu = ['Home',"NLP(files)",'About']
#     choice = st.sidebar.selectbox('Menu',menu)

#     if __name__== '__main__':

#         main()



import streamlit as st
import streamlit.components.v1 as stc
#load EDA package 
import pandas as pd 

#load nlp package 
import spacy
nlp = spacy.load('en_core_web_sm')

#text cleaning packgs
import neattext as nt
import neattext.functions as nfx

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import string
import nltk
nltk.download('vader_lexicon')
from spacy import displacy
from collections import Counter
nltk.download('stopwords')
nltk.download('punkt')
#from  textblob import Textblob

def text_analyzer(my_text):

    docx = nlp(my_text)
    alldata = [(token.text,token.shape_,token.pos,token.tag_,token.lemma_,token.is_alpha,token.is_stop) for token in docx]
    df = pd.DataFrame(alldata,columns = ['Token','Shape','Pos','Tag','Lemma','IsAplha','Is_Stopword'])
    return df

def get_entities(my_text):
    docx = nlp(my_text)
    entities = [(entity.text,entity.label_) for entity in docx.ents]
    return entities

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""



def render_entities(rawtext):
    docx = nlp(rawtext)
    html = displacy.render(docx,style = 'ent')
    html = html.replace('\n\n','\n')
    result = HTML_WRAPPER.format(html)
    return result


#Fnc to get most common tokens

def get_most_common_tokens(my_text,num = 4):
    word_tokens = Counter(my_text.split())
    most_common_tokens = dict(word_tokens.most_common(num))
    return most_common_tokens

#func to get sentiment 


def get_sentiment(my_text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(my_text)

    # Extracting the compound score, which represents the overall sentiment
    compound_score = sentiment_scores['compound']

    # Assigning sentiment labels based on the compound score
    if compound_score >= 0.05:
        sentiment_label = 'Positive'
    elif compound_score <= -0.05:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'

    return {
        'compound_score': compound_score,
        'sentiment_label': sentiment_label
    }





def appilcation():
    st.write('NLP APP With Streamlit')
    menu = ['Home', 'NLP (files)', 'About']
    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Home':
        st.subheader("Home: Analysis Text")
        raw_text = st.text_area('Enter your text')
        num_of_most_common = st.sidebar.number_input("Most Common Tokens",5,15)
        if st.button('Analyse'):
            with st.expander('Original Text'):
                #st.write('raw_tesxt')
                 st.write(raw_text)
            with st.expander("Text analysis"):
                 token_result_df = text_analyzer(raw_text)
                 st.dataframe(token_result_df)
            with st.expander('Entities'):
                #    entity_result = get_entities(raw_text)
                #    st.table(entity_result) 
                   entity_result = render_entities(raw_text)
                   stc.html(entity_result,height =100,scrolling = True)

            #column Layout

            col1,col2 = st.columns(2)

            with col1:
                 with  st.expander("Word Stats"):
                    #  st.write('cdhdbcwdbc')
                     st.info('Word statistics')
                     docx = nt.TextFrame(raw_text)
                     st.write(docx.word_stats())

                 with st.expander('Top keywors'):
                     #st.write('acadc')
                     st.info('Top Keywords/tokens')
                    # process_text = nfx.remove_stopwards(raw_text)
                     words = word_tokenize(raw_text)
                     filtered_words = [word for word in words if word.lower() not in stopwords.words('english')]
                     process_text = ' '.join(filtered_words)
                     Keywords = get_most_common_tokens(process_text)
                     st.write(Keywords)
                    
                 with st.expander('Sentiment'):
                      
                    sentiment_result = get_sentiment(raw_text)
                    st.write(sentiment_result)


            with col2:
                 with  st.expander("Plot word Frequency"):
                     st.write('cdhdbcwdbc')
                     
                 with st.expander('Plot parts Of Speech'):
                     st.write('acadc')
                    
                 with st.expander('Plot word Cloud'):
                     st.write('dcdcwde')
                  
    elif choice == "NlP(files)":
        st.subheader('NLP Task')
    else:
        st.subheader('About')



if __name__ == '__main__':
    appilcation()
