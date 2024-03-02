
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
                    'Lê Duy Quang - Trịnh Việt Hoàng  \n'
                    'Nguyễn Đức Hậu - Tạ Hữu Huy Hoàng ' 
                    )
st.sidebar.image('banner2.jpg', use_column_width=True)
st.sidebar.markdown('# Chọn game bạn muốn!  \n'
                    'Ứng dụng sẽ gợi ý cho bạn 5 game có nội dung tương tự!')
st.sidebar.markdown('')
ph = st.sidebar.empty()
selected_game = ph.selectbox('Chọn 1 trong 787 game của Nintendo Switch '
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
    st.markdown('Hệ thống gợi ý này sử dụng những thuật toán dựa trên '
                'các kĩ thuật học không giám sát.')

    # Phần cào dữ liệu
    st.markdown('## Web scraping')
    st.text('')
    st.markdown('Tập dữ liệu được lấy từ wikipedia:')
    st.markdown('* https://en.wikipedia.org/wiki/List_of_Nintendo_Switch_games_(Q%E2%80%93Z)')
    st.markdown('Cào dữ liệu từ bảng có chứa đường link đến từng game. Sau đó, '
                'với mỗi đường link, chúng mình cào thêm dữ liệu về gameplay, nội dung, hoặc cả 2. '
                'Sau đó chúng mình tạo ra được dataframe:')
    games_df
    st.markdown('Sử dụng [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/), '
                'để thu thập văn bản:')
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
    st.markdown('Từ ma trận này, chúng mình tạo thành 1 dataframe:')
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
        st.markdown('> _Bạn có trên tay Nintendo Switch, phá đảo 1 con game thú vị '
                    'và muốn gợi ý những game tương tự?_')
        st.text('')
        st.markdown("Trang web này sẽ gợi ý cho bạn 5 game của Nintendo "
                    'dựa trên nội dung, gameplay và những điều tương đồng khác để bạn chọn lựa!')
        st.markdown('Thuật toán dựa trên xử lý ngôn ngữ tự nhiên và kĩ thuật học không giám sát \n'
                    ' Bấm *__Chi tiết__* để biết thêm!')
        st.text('')
        st.warning(':point_left: Chọn 1 game từ menu!')
        st.markdown('Một số game nổi bật: ')
        st.text('')
        st.markdown("[![Game1](https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_801/b_white/f_auto/q_auto/ncom/software/switch/70010000063714/fb30eab428df3fc993b41c76e20f72e4d76d49734d17d31996b5ab61c414b117)](https://www.nintendo.com/us/store/products/the-legend-of-zelda-tears-of-the-kingdom-switch/)")
        st.markdown("[![Game2](https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_801/b_white/f_auto/q_auto/ncom/software/switch/70010000001107/fc38578b345af43fbc7bd1eb5be84594ab6c28bd8927ce8fca97e08071f2705d)](https://www.nintendo.com/us/store/products/bayonetta-2-switch/)")
        st.markdown("[![Game3](https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_801/b_white/f_auto/q_auto/ncom/software/switch/70010000058797/f37d3c4f766c56db53734e7b74328b66d2a1d9cf37257dcc623435862b145bad)](https://www.nintendo.com/us/store/products/kirbys-return-to-dream-land-deluxe-switch/)")
        st.markdown("[![Game4](https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_801/b_white/f_auto/q_auto/ncom/software/switch/70010000001130/c42553b4fd0312c31e70ec7468c6c9bccd739f340152925b9600631f2d29f8b5)](https://www.nintendo.com/us/store/products/super-mario-odyssey-switch/)")
        st.markdown("[![Game5](https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_801/b_white/f_auto/q_auto/ncom/software/switch/70010000053971/842b2784d91520d41a947dec17fac116fec889bb1f1db4023615af8429dae00d)](https://www.nintendo.com/us/store/products/pokemon-violet-switch/)")
        st.markdown("[![Game6](https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_801/b_white/f_auto/q_auto/ncom/software/switch/70010000053336/e933b48650b33b355e9cf2583da5c94b77180e40fb02d050041083dd62f4df39)](https://www.nintendo.com/us/store/products/xenoblade-chronicles-3-switch/)")
        st.markdown("[![Game7](https://assets.nintendo.com/image/upload/ar_16:9,c_lpad,w_801/b_white/f_auto/q_auto/ncom/en_US/games/switch/n/nier-automata-the-end-of-yorha-edition-switch/hero)](https://www.nintendo.com/us/store/products/nier-automata-the-end-of-yorha-edition-switch/)")

            
