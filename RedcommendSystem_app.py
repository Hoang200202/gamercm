
import streamlit as st
import pandas as pd
import textwrap

# Load and Cache the data
@st.cache_data(persist=True)
def getdata():
    games_df = pd.read_csv("D:/N4HK2/HTGY/GameRCM/datasets-20240125T131440Z-001/datasets/Games_dataset.csv", index_col=0)
    similarity_df = pd.read_csv("D:/N4HK2/HTGY/GameRCM/datasets-20240125T131440Z-001/datasets/sim_matrix.csv", index_col=0)
    return games_df, similarity_df

games_df, similarity_df = getdata()[0], getdata()[1]

# Sidebar
st.sidebar.markdown('__Nintendo Switch game recommender__  \n Bài tập của nhóm 5  \n'
                    'Nông Minh Đức - Trịnh Việt Hoàng')
st.sidebar.image('D:/N4HK2/HTGY/GameRCM/images-20240125T131411Z-001/images/banner.png', use_column_width=True)
st.sidebar.markdown('# Chọn game của bạn!')
st.sidebar.markdown('')
ph = st.sidebar.empty()
selected_game = ph.selectbox('Chọn 1 trong 787 game của Nintendo '
                             'từ menu: (bạn có thể nhập tên game ở đây)',
                             [''] + games_df['Title'].to_list(), key='default',
                             format_func=lambda x: 'Select a game' if x == '' else x)

st.sidebar.markdown("# Want to know what's behind this app?")
st.sidebar.markdown("Click on the button :point_down:")
btn = st.sidebar.button("How this app works?")

# Explanation with button 
if btn:
    selected_game = ph.selectbox('Select one among the 787 games ' \
                                 'from the menu: (you can type it as well)',
                                 [''] + games_df['Title'].to_list(),
                                 format_func=lambda x: 'Select a game' if x == '' else x,
                                 index=0, key='button')

    st.markdown('# How does this app work?')
    st.markdown('---')
    st.markdown('The recommendation system used in this app employs a series of algorithms based '
                'on unsupervised learning techniques.')

    # Scraping
    st.markdown('## Web scraping')
    st.text('')
    st.markdown('Beforehand, the dataset was obtained by scraping these two Wikipedia pages:')
    st.markdown('* https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(Q%E2%80%93Z)')
    st.markdown('I scraped the table entries which contain links to their video game pages. Then, '
                'for each video game, I scraped either the Gameplay section, the Plot section, or both. '
                'With this, I created the following dataframe:')
    games_df
    st.markdown('Using [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/), '
                'the text scraping looks like this:')
    st.code("""
text = ''
    
for section in soup.find_all('h2'):
        
    if section.text.startswith('Game') or section.text.startswith('Plot'):

        text += section.text + ''

        for element in section.next_siblings:
            if element.name and element.name.startswith('h'):
                break

            elif element.name == 'p':
                text += element.text + ''

    else: pass
    """, language='python')

    # Text Processing
    st.markdown('## Text Processing')
    st.markdown('Using [NLTK](https://www.nltk.org) for natural language processing, I defined '
                'the rules for tokenizing and stemming the plots.')
    st.code(""" 
def tokenize_and_stem(text):
    
    # Tokenize by sentence, then by word
    tokens = [word for sent in nltk.sent_tokenize(text) 
              for word in nltk.word_tokenize(sent)]
    
    # Filter out raw tokens to remove noise
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    
    # Stem the filtered_tokens
    stems = [stemmer.stem(word) for word in filtered_tokens]
    
    return stems    
    """, language='python')

    # Vectorizing
    st.markdown('## Text vectorizing')
    st.markdown('I employ a [TF-IDF vectorizer](https://towardsdatascience.com/'
                'natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76) '
                '(term frequency - inverse document frequency) to translate words into vectors. '
                'The plots are then fit/transformed using this vectorizer. Along the way, '
                'the `tokenize_and_stem` function is used, as well as a stop words remover.')
    st.code("""
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem,
                                 ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in games_df["Plots"]])
    """, language='python')

    # Similarity distance
    st.markdown('## Similarity distance')
    st.markdown('Finally, the similarity distance of two texts is computed by substracting the '
                'cosine of the two associated vectors from 1:')
    st.code("""similarity_distance = 1 - cosine_similarity(tfidf_matrix)""", language='python')
    st.markdown('From this matrix, we can create a dataframe:')
    similarity_df
    st.markdown('Then, once a game is selected, we query the top 5 most similar games '
                'according to this table.')

# Recommendations
if selected_game:

    link = 'https://en.wikipedia.org' + games_df[games_df.Title == selected_game].Link.values[0]

    # DF query
    matches = similarity_df[selected_game].sort_values()[1:6]
    matches = matches.index.tolist()
    matches = games_df.set_index('Title').loc[matches]
    matches.reset_index(inplace=True)

    # Results
    cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

    st.markdown("# The recommended games for [{}]({}) are:".format(selected_game, link))
    for idx, row in matches.iterrows():
        st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))
        st.markdown(
            '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 600)[0], row['Link']))
        st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
        st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))

else:
    if btn:
        pass
    else:
        st.markdown('# Nintendo Switch game recommender')
        st.text('')
        st.markdown('> _So you have a Nintendo Switch, just finished an amazing game, and would like '
                    'to get recommendations for similar games?_')
        st.text('')
        st.markdown("This app lets you select a game from the dropdown menu and you'll get five "
                    'recommendations that are the closest to your game according to the gameplay and/or plot.')
        st.markdown('The algorithm is based on natural language processing and unsupervised learning '
                    'techniques &#151; click on the *__How this app works?__* button to know more!')
        st.text('')
        st.warning(':point_left: Select a game from the dropdown menu!')
