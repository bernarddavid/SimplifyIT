# Python script for SimplifyIT web app

# Import streamlit, which is package to build webapp
import streamlit as st

# Import other packages
# Load packages
import pandas as pd
import numpy as np

import dill
import sklearn

import spacy
import textstat
import gpt_2_simple as gpt2
import tensorflow as tf

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec, Doc2Vec

# Specify location of trained models
TrnMod = '../trained_models/'


#############################
# Text difficulty functions #
#############################

# Load text difficulty model
text_diff_mod = dill.load(open(TrnMod+'text_difficulty.pickle', 'rb'))

# Define functions used 
# Function splits text, t, into sentences
def sent_break(t):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(t)
    return doc.sents

# Function counts the number of words in text t
def word_count(t):
    sents = sent_break(t)
    n_words = 0
    for s in sents:
        n_words += len([token for token in s])
    return n_words

# Function counts the number of sentences in text t
def sent_count(t):
    sents = sent_break(t)
    return len(list(sents))

# Function evaluates text difficulty: input is text and output is a difficulty level (advanced, elementary, intermediate)
def text_difficulty(t):
    t = [t] # Put text to list
    df = pd.DataFrame(t, columns = ['text']) # Initiate dataframe with text
    df['difficulty'] = df['text'].apply(textstat.flesch_reading_ease) # Flesh reading difficulty
    df['n_sent'] = df['text'].apply(sent_count) # Number of sentences
    df['n_syll'] = df['text'].apply(textstat.syllable_count) # Number of syllables
    df['n_lex'] = df['text'].apply(textstat.lexicon_count) # Number of words
    df['lex_sent'] = df['n_lex']/df['n_sent'] # Word to sentence ratio
    df['syll_lex'] = df['n_syll']/df['n_lex'] # Syllable to word ratio
    df = df[['difficulty', 'lex_sent', 'syll_lex', 'n_syll']] # Reduce to 
    
    level = text_diff_mod.predict(df)[0] # Predict reading difficulty of text
    
    diff_dict = {0:'advanced', 1:'elementary', 2:'intermediate'} # Dictionary stores descriptors of levels
    
    return diff_dict[level]

#############################
# Text similarity functions #
#############################
    
# Load Doc2Vec similarity model
similarity_mod = Doc2Vec.load(TrnMod + "d2v.model")

# Function takes a list of words and returns a list in which all stop words are removed
def remove_stop_words(wordlist):
    # Get all English stop words
    stops = set(stopwords.words("english"))  
    nostops = [w for w in wordlist if w not in stops]
    return nostops

# Function takes a list of words and returns a list of word stems
def stem_words(wordlist):
    # Initialize object to stem words
    ps = PorterStemmer()
    stems = [ps.stem(w) for w in wordlist]
    return stems

def text_similarity(t1, t2):
    t1, t2 = str(t1), str(t2) # Convert text entries to strings
    # Initialize data frame to clean and evaluate text 
    t = [t1, t2]
    df = pd.DataFrame(t, columns = ['text'])
    
    # Prep texts for Doc2Vec similarity
    df['text_c'] = df['text'].str.replace(r'[^a-zA-Z\s+]', '').str.lower() # Remove numbers and symbols and convert string to lower
    df['text_c'] = df['text_c'].str.replace('\n', '')
    df['text_c'] = df['text_c'].str.replace(r'\s+\s+', ' ') # Replace double spaces with single space
    
    df['text_c'] = df['text_c'].str.replace('mss', '')
    
    df['text_c'] = df['text_c'].apply(word_tokenize) # Tokenize text entries
    df['text_c'] = df['text_c'].apply(stem_words) # Stem text entries
    df['text_c'] = df['text_c'].apply(remove_stop_words) # Remove stop words
    
    # Calculate cosine simlarity of both sentences
    cos_sim = similarity_mod.wv.n_similarity(df['text_c'][0], df['text_c'][1])
    
    return cos_sim
    
##############################################
# Function to identify best fitting sentence #
##############################################

def sentence_fit(gen_text, orig_text):
    df = pd.DataFrame(gen_text, columns = ['generated']) # Text generated from GPT2 stored in dataframe
    df['generated'] = df['generated'].str.replace(r' +,', ',').str.replace(r' +\.', '.') # Remove spaces in front of punctuation
    df['similarity'] = df['generated'].apply(lambda x: text_similarity(orig_text, x)) # Assess cosine similarity betweeen sentences
    df['n_syll'] = df['generated'].apply(textstat.syllable_count) # Count number of syllables
    df['n_lex'] = df['generated'].apply(textstat.lexicon_count) # Count number of words
    df['syll_lex'] = df['n_syll']/df['n_lex'] # Syllable to word ratio
    
    # Flags to indicate whether generated text has fewer words, syallables, or syll to word ratio
    df['rel_syll'] = np.where(df['n_syll'] < textstat.syllable_count(orig_text), 1, 0)
    df['rel_lex'] = np.where(df['n_lex'] < textstat.lexicon_count(orig_text), 1, 0)
    df['rel_rat'] = np.where(df['syll_lex'] < textstat.syllable_count(orig_text)/textstat.lexicon_count(orig_text), 1, 0)
    
    # Sum binary indicators of relative sentence simplicity
    df['rel_simp'] = (df['rel_syll'] + df['rel_lex'] + df['rel_rat'])/3
    
    # Fit score is weighted sum of similarity and relative sentence simplicity
    # Highest score will be chosen
    df['fit_score'] = 0.7*df['similarity'] + 0.3*df['rel_simp']
    
    # Subset data and rename columns
    df['Original'] = orig_text
    df = df[['Original', 'generated', 'similarity', 'rel_simp', 'fit_score']]
    df.columns = ['Original', 'Generated', 'Similarity', 'Simplicity', 'Fit Score']
    
    return df


#################################
# Function to generate new text #
#################################

# Function takes model specification for SimpleGPT2 model and input text then returns new text
@st.cache
def generate_text(size, mod_dir, ft_dir, ft_dat, n_steps, input_text, n_new_sent):
    # Simple GPT2 specifications
    gpt2size = size
    gpt2dir = mod_dir
    loaddir = ft_dir
    ft_mod = ft_dat + '_' + gpt2size + '_' + str(n_steps)
    
    if gpt2size == '355M':
        sent_delim = '|<EndSentence1>|'
    if gpt2size == '124M':
        sent_delim = '<|textdelim|>' # Used different delimeters based on trained model
    
    # Initiate tf session and load fine-tuned GPT2 model
    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess,
                  run_name = ft_mod,
                  checkpoint_dir = loaddir)
    
    # Split input text into list of sentences
    sent_list = [s for s in sent_break(InputText)]
    
    # Initialize dataframes to return
    all_new_sent = pd.DataFrame()
    best_new_sent = pd.DataFrame()
    
    x = 0 # Start counter
    for s in sent_list:
        x += 1
        orig_text = str(s).strip()
    
        gen_text = gpt2.generate(sess, 
                                 nsamples = n_new_sent,
                                 prefix = '<|startoftext|>' + orig_text + sent_delim,
                                 truncate = '<|endoftext|>',
                                 include_prefix = False,
                                 run_name = ft_mod,
                                 checkpoint_dir = loaddir,
                                 return_as_list = True,
                                 temperature = 0.8)

        AllSent = sentence_fit(gen_text, orig_text).reset_index(drop = True)
        AllSent['SentNo'] = 'Sent'+str(x)
        AllSent = AllSent.drop_duplicates()
        AllSent = AllSent[(AllSent['Generated'] != AllSent['Original']) & (AllSent['Similarity'] > 0.9)]
        
        Sent = list(AllSent[(AllSent['Fit Score'] == AllSent['Fit Score'].max())]['Generated'])[0]

        # Data frames with best fitting sentences and all sentences
        all_new_sent = all_new_sent.append(AllSent)
        best_new_sent = best_new_sent.append(pd.DataFrame({'SentNo': 'Sent'+str(x), 
                                                           'Original' : [orig_text],
                                                           'Generated' : [Sent]}))

    return all_new_sent.reset_index(drop = True), best_new_sent.reset_index(drop = True)

##########################
# Begin code for webpage #
##########################

# Title of app
st.title('SimplifyIT')

# App tagline
st.text('Don\'t let language keep you from learning. SimplifyIT!')

# Instructions for app use
st.markdown('''**SimplifyIT**: This app reduces the linguistic complexity of informational text. Simply copy and paste the text you want to simplify into the grey text box, and we'll take care of the rest!''')

# Store text entered by user
InputText = st.text_area('Text to simplify:')

# Get text difficulty
level = text_difficulty(InputText)

return_string = 'This text is written for **' + level + '** readers.'

st.markdown(return_string)

# GPT2 Model specification that are fed into "generate text" function
gpt2size = '124M' # Either '124M' or '355M'
n_steps = 2000 # Number of steps used to train model
gpt2dir = '../gpt2models' # Location where pre-trained gpt2 models stored
loaddir = '../trained_models/checkpoint' # Location where fine-tuned gpt2 models stored
ft_dat = 'wiki_sentence' # Data set on which GPT2 trained

st.markdown('Generating new text...')

# Generate replacement text from fine-tuned gpt2 model
all_new, best_new = generate_text(size = gpt2size, 
                                  mod_dir = gpt2dir, 
                                  ft_dir = loaddir, 
                                  ft_dat = ft_dat, 
                                  n_steps = n_steps, 
                                  input_text =  InputText, 
                                  n_new_sent = 10)

# Set to display all text in pandas dataframe
pd.set_option('max_colwidth', None)

# Merge simplified sentences into paragraph
simplified_text = ' '.join(best_new['Generated'])

simp_lev = text_difficulty(simplified_text)

# Display new text
st.markdown('Here is your simplified text:')
st.markdown('_'+simplified_text+'_')
st.markdown('The new text is written for **' + simp_lev + '** readers.')

# Ask for user satisfaction about new text
satis = st.radio(
    'What do you think about this simplification?',
    ('It\'s great, I love it!', 
     'I\'d  like to make some modifications.')
)

# Replies to satisfaction options
if satis == 'It\'s great, I love it!':
    st.markdown('Fantastic, happy learning!')
if satis == 'I\'d  like to make some modifications.':
    st.markdown('Okay, let\'s find some alternatives.') 
    st.markdown('Which sentences would you like to modify?')
    
    # If dissatisfied, user selects which sentence(s) to be changed.
    sent_review = st.multiselect('Select all of the sentences you want to change.', best_new['Generated'])
    
    # For each sentence with which user is dissatisfied, allow user to choose replacement.
    sent_select = pd.DataFrame()
    for i in sent_review:
        # Get sentence number of replaced sentence
        sent_no = list(all_new[all_new['Generated'] == i]['SentNo'])[0]
        
        st.markdown('You are dissatisfied with this simplification: \"*' + str(i) + '*\"')
        replace2 = st.selectbox('Choose one of the sentences below as a replacement:', 
                                 all_new[all_new['SentNo'] == sent_no]['Generated'].reset_index(drop = True),
                                 key = sent_no)
        
        sent_select = sent_select.append(pd.DataFrame({'SentNo' : [sent_no],
                                                       'Generated' : [replace2]}))
        
    sent_select_ret = best_new[~best_new['SentNo'].isin(sent_select['SentNo'])][['SentNo', 'Generated']]
    sent_select_ret = sent_select_ret.append(sent_select).sort_values(by = 'SentNo').reset_index(drop = True)
    selected_text = ' '.join(sent_select_ret['Generated'])
    
    st.markdown('Here is the text with your requested revisions:')
    st.markdown('_' + selected_text + '_')
    
    
    
    