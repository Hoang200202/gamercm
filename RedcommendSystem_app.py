
import streamlit as st
import pandas as pd
import textwrap

# Load and Cache the data
@st.cache_data(persist=True)
def getdata():
    games_df = pd.read_csv("Games_dataset.csv", index_col=0)
    similarity_df = pd.read_csv("sim_matrix.csv", index_col=0)
    return games_df, similarity_df

games_df, similarity_df = getdata()[0], getdata()[1]

# Sidebar
st.sidebar.markdown('__Nintendo Switch game recommender__  \n Bài tập của nhóm 4  \n'
                    'Nông Minh Đức - Trịnh Việt Hoàng  \n'
                    )
st.sidebar.image('banner2.jpg', use_column_width=True)
st.sidebar.markdown('# Chọn game của bạn!')
st.sidebar.markdown('')
ph = st.sidebar.empty()
selected_game = ph.selectbox('Chọn 1 trong 787 game của Nintendo '
                             'từ menu: (bạn có thể nhập tên game ở đây)',
                             [''] + games_df['Title'].to_list(), key='default',
                             format_func=lambda x: 'Select a game' if x == '' else x)

st.sidebar.markdown("# More info?")
st.sidebar.markdown("Bấm nút dưới đây để tìm hiểu về app của chúng mình")
btn = st.sidebar.button("Chi tiết")

# Giải thích về các nút 
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

    # Phần cào dữ liệu
    st.markdown('## Web scraping')
    st.text('')
    st.markdown('Tập dữ liệu được lấy từ wikipedia:')
    st.markdown('* https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(Q%E2%80%93Z)')
    st.markdown('Cào dữ liệu từ bảng có chứa đường link đến từng game. Sau đó, '
                'với mỗi đường link, chúng mình cào thêm dữ liệu về gameplay, nội dung, hoặc cả 2. '
                'Sau đó chúng mình tạo ra được dataframe:')
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

    # Xử lý văn bản
    st.markdown('## Text Processing')
    st.markdown('Sử dụng [NLTK](https://www.nltk.org) để xử lý ngôn ngữ tự nhiên, '
                'chuẩn hóa dữ liệu văn bản với tokenizing.')
    st.code(""" 
def tokenize_and_stem(text):
    
    # Token hóa với câu rồi đến từng chữ
    tokens = [word for sent in nltk.sent_tokenize(text) 
              for word in nltk.word_tokenize(sent)]
    
    # Khử nhiễu
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    
    # Tối giản hóa
    stems = [stemmer.stem(word) for word in filtered_tokens]
    
    return stems    
    """, language='python')

    # Vector hóa
    st.markdown('## Text vectorizing')
    st.markdown('Chúng mình dùng [TF-IDF vectorizer](https://towardsdatascience.com/'
                'natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76) '
                '(term frequency - inverse document frequency) để vector hóa văn bản. '
                'Nội dung được vector hóa theo cách này. Theo đó, '
                'hàm `tokenize_and_stem` sẽ được sử dụng, và loại bỏ các stop words.')
    st.code("""
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem,
                                 ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in games_df["Plots"]])
    """, language='python')

    # Tính được khoảng cách độ tương đồng (Similarity Distance)
    st.markdown('## Similarity distance')

    st.code("""similarity_distance = 1 - cosine_similarity(tfidf_matrix)""", language='python')
    st.markdown('From this matrix, we can create a dataframe:')
    similarity_df
    st.markdown('Sau đó, khi đã chọn được game, ứng dụng sẽ gợi ý ra 5 game tương đồng '
                'thông qua bảng này.')

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
        st.markdown('> _Bạn có trên tay Nintendo Switch, chơi xong 1 con game thú vị '
                    'và muốn gợi ý 1 game tương tự?_')
        st.text('')
        st.markdown("Trang web này sẽ gợi ý cho bạn 5 game của Nintendo "
                    'dựa trên nội dung, gameplay và những điều tương đồng khác để bạn chọn lựa!')
        st.markdown('Thuật toán dựa trên xử lý ngôn ngữ tự nhiên và kĩ thuật học không giám sát '
                    '; bấm *__Chi tiết?__* để biết thêm!')
        st.text('')
        st.warning(':point_left: Chọn 1 game từ menu!')
        st.markdown('Một số game nổi bật: ')
        st.text('')
        col1, col2 = st.beta_columns(2)
        with col1:
            st.markdown("[![Game1](https://upload.wikimedia.org/wikipedia/en/c/c6/The_Legend_of_Zelda_Breath_of_the_Wild.jpg)](https://en.wikipedia.org/wiki/The_Legend_of_Zelda:_Breath_of_the_Wild)")
            st.markdown("[![Game2](https://upload.wikimedia.org/wikipedia/en/b/b1/Bayonetta_2_box_artwork.png)](https://en.wikipedia.org/wiki/Bayonetta_2)")
            st.markdown("[![Game3](https://upload.wikimedia.org/wikipedia/en/6/65/Kirby_Star_Allies.jpg)](https://en.wikipedia.org/wiki/Kirby_Star_Allies)")
            st.markdown("[![Game4](https://upload.wikimedia.org/wikipedia/en/5/50/Super_Smash_Bros._Ultimate.jpg)](https://en.wikipedia.org/wiki/Super_Smash_Bros._Ultimate)")
            st.markdown("[![Game5](https://upload.wikimedia.org/wikipedia/en/thumb/3/3e/Pokemon_Brilliant_Diamond_Shining_Pearl.png/330px-Pokemon_Brilliant_Diamond_Shining_Pearl.png)](https://en.wikipedia.org/wiki/Pok%C3%A9mon_Brilliant_Diamond_and_Shining_Pearl)")
            st.markdown("[![Game6](https://upload.wikimedia.org/wikipedia/en/7/76/Xenoblade_3.png)](https://en.wikipedia.org/wiki/Xenoblade_Chronicles_3)")
            st.markdown("[![Game7](https://upload.wikimedia.org/wikipedia/en/8/8a/Fitness_Boxing.jpg)](https://en.wikipedia.org/wiki/Fitness_Boxing)")
        with col2:
            st.write("The Legend of Zelda: Breath of the Wild[b] is a 2017 action-adventure game developed and published by Nintendo for the Nintendo Switch and Wii U. Set at the end of the Zelda timeline, the player controls an amnesiac Link as he sets out to save Princess Zelda and prevent Calamity Ganon from destroying the world. Players explore the open world of Hyrule while they collect items and complete objectives such as puzzles or side quests. Breath of the Wild's world is unstructured and encourages exploration and experimentation; the story can be completed in a nonlinear fashion.")
