import sqlite3
import streamlit as st
import pandas as pd
import textwrap
import pickle
from pathlib import Path
import streamlit_authenticator as stauth


# --------USER AUTHENTICATION--------------
names = ["Trinh Hoang", "Nong Duc"]
usernames = ["tvhoang", "nmduc"]

# load hashed passwords
file_path =Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "game_rcm", "hoangvaduc", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Tên đăng nhập hoặc mật khẩu bị sai!")
if authentication_status == None:
    st.warning("Mời bạn đăng nhập!")
if authentication_status:


    # Load and Cache the data
    @st.cache_data(persist=True)
    def getdata():
        conn = sqlite3.connect('GameData')
        query = "SELECT * FROM Game_data"
        games_df = pd.read_sql_query(query, conn, index_col=None)
        games_df = games_df.rename(columns={games_df.columns[1]: 'Title'})
        games_df = games_df.rename(columns={'Unnamed: 0': 'Index'})
        conn.close()
        
        similarity_df = pd.read_csv("sim_matrix.csv", index_col=0)
        return games_df, similarity_df
    
    games_df, similarity_df = getdata()[0], getdata()[1]
    
    # Sidebar
    authenticator.logout("Log out","sidebar")
    st.sidebar.title(f"Welcome {name}")
    st.sidebar.markdown('__Nintendo Switch game recommender__  \n Bài tập của nhóm 4  \n'
                
                        'Nông Minh Đức - Trịnh Việt Hoàng' 
                        )
    st.sidebar.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTUNyyuOJWHq1SBTlPb485NYffNXxljjEeOVCWYQnhPBSC9YwFIAgkZVl_EqnFAD31Hexg&usqp=CAU', use_column_width=True)
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
        st.markdown('Hệ thống gợi ý  sử dụng những thuật toán dựa trên '
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
                    'Nội dung được vector hóa theo cách . Theo đó, '
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
        st.markdown('Từ ma trận , chúng mình tạo thành 1 dataframe:')
        similarity_df
        st.markdown('Sau đó, khi đã chọn được game, ứng dụng sẽ gợi ý ra 5 game tương đồng '
                    'thông qua bảng .')
    
    # Recommendations

if selected_game:
  
    link = 'https://en.wikipedia.org' + games_df[games_df.Title == selected_game].Link.values[0]
    plots= games_df[games_df.Title == selected_game].Plots.values[0]
    
    # DF query
    matches = similarity_df[selected_game].sort_values()[1:6]
    matches = matches.index.tolist()
    matches = games_df.set_index('Title').loc[matches]
    matches.reset_index(inplace=True)

    # Results
    cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

    st.markdown("# The recommended games for [{}]({}): \n {} ".format(selected_game, link, plots))
    for idx, row in matches.iterrows():
        st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

        st.markdown(
            '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
        st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
        st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))

        # Hiển thị ảnh từ đường liên kết trong cột 'image'
       

else:
    if btn:
        pass
    else:
            st.markdown('# Website giới thiệu và gợi ý game')
            st.text('')
            st.markdown('> _Bạn có trên tay Nintendo Switch, phá đảo 1 con game thú vị '
                        'và muốn gợi ý những game tương tự?_')
            st.text('')
            st.markdown("Trang web  sẽ gợi ý cho bạn 5 game của Nintendo "
                        'dựa trên nội dung, gameplay và những điều tương đồng khác để bạn chọn lựa!')
            st.markdown('Thuật toán dựa trên xử lý ngôn ngữ tự nhiên và kĩ thuật học không giám sát \n'
                        ' Bấm *__Chi tiết__* để biết thêm!')
            st.text('')
            st.warning(':point_left: Chọn 1 game từ menu!')
            st.markdown('Một số game nổi bật: ')
            st.text('')
            cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']
            
            
            st.title(games_df.Title[2])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/a/aa/198X_Nintendo.com_Hero_Image.jpg/220px-198X_Nintendo.com_Hero_Image.jpg")
            st.write('Nội dung chính:', games_df.Plots[2][:200],'...')
            row_data = games_df.loc[2, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn 198X"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[2]
                plots= games_df.Plots.values[2]
                
                # DF query
                matches = similarity_df[games_df.Title[2]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những game gợi ý cho game [{}]({}): \n {} ".format(games_df.Title[2], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            st.title(games_df.Title[1])
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/140_%28video_game%29_Logo.svg/220px-140_%28video_game%29_Logo.svg.png")
            st.write('Nội dung chính:', games_df.Plots[1][:200],'...')
            row_data = games_df.loc[1, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn 140"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[1]
                plots= games_df.Plots.values[1]
                
                # DF query
                matches = similarity_df[games_df.Title[1]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[1], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            
            
            st.title(games_df.Title[3])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/45/20XX_%28video_game%29.jpg/220px-20XX_%28video_game%29.jpg")
            st.write('Nội dung chính:', games_df.Plots[3][:200],'...')
            row_data = games_df.loc[3, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn 20XX"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[3]
                plots= games_df.Plots.values[3]
                
                # DF query
                matches = similarity_df[games_df.Title[3]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[3], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            st.title(games_df.Title[4])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/f/ff/6180TheMoonSteamHeader.jpg/220px-6180TheMoonSteamHeader.jpg")
            st.write('Nội dung chính:', games_df.Plots[4][:200],'...')
            row_data = games_df.loc[4, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn 6180 The Moon"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[4]
                plots= games_df.Plots.values[4]
                
                # DF query
                matches = similarity_df[games_df.Title[4]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[4], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            
            st.title(games_df.Title[5])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/c0/7_Billion_Humans.jpg/220px-7_Billion_Humans.jpg")
            st.write('Nội dung chính:', games_df.Plots[5][:200],'...')
            row_data = games_df.loc[5, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Billion Human"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[5]
                plots= games_df.Plots.values[5]
                
                # DF query
                matches = similarity_df[games_df.Title[5]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[5], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            
            st.title(games_df.Title[6])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/9/91/99_vidas.jpg/220px-99_vidas.jpg")
            st.write('Nội dung chính:', games_df.Plots[6][:200],'...')
            row_data = games_df.loc[6, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Vidas"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[6]
                plots= games_df.Plots.values[6]
                
                # DF query
                matches = similarity_df[games_df.Title[6]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[6], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            st.title(games_df.Title[7])
            st.image("https://upload.wikimedia.org/wikipedia/en/1/12/Aaero.jpg")
            st.write('Nội dung chính:', games_df.Plots[7][:200],'...')
            row_data = games_df.loc[7, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Aero"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[7]
                plots= games_df.Plots.values[7]
                
                # DF query
                matches = similarity_df[games_df.Title[7]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[7], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
                    
     
            st.title(games_df.Title[8])
            st.image("https://upload.wikimedia.org/wikipedia/en/7/73/Abzu.jpg")
            st.write('Nội dung chính:', games_df.Plots[8][:200],'...')
            row_data = games_df.loc[8, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Abzu"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[8]
                plots= games_df.Plots.values[8]
                
                # DF query
                matches = similarity_df[games_df.Title[8]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[8], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
                    
            
            st.title(games_df.Title[9])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/43/Aegis_Defenders.jpg/220px-Aegis_Defenders.jpg")
            st.write('Nội dung chính:', games_df.Plots[9][:200],'...')
            row_data = games_df.loc[9, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Aegis Defenders"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[9]
                plots= games_df.Plots.values[9]
                
                # DF query
                matches = similarity_df[games_df.Title[9]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[9], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            st.title(games_df.Title[10])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/0/03/Afterparty_cover_art.jpg/220px-Afterparty_cover_art.jpg")
            st.write('Nội dung chính:', games_df.Plots[10][:200],'...')
            row_data = games_df.loc[10, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn After Party"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[10]
                plots= games_df.Plots.values[10]
                
                # DF query
                matches = similarity_df[games_df.Title[10]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[10], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            st.title(games_df.Title[11])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/6/63/Agony_2017_pre-release_Steam.jpg/220px-Agony_2017_pre-release_Steam.jpg")
            st.write('Nội dung chính:', games_df.Plots[11][:200],'...')
            row_data = games_df.loc[11, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Agony"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[11]
                plots= games_df.Plots.values[11]
                
                # DF query
                matches = similarity_df[games_df.Title[11]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[11], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
                    
            st.title(games_df.Title[12])
            st.image("https://upload.wikimedia.org/wikipedia/en/6/6e/Alien_Isolation.jpg")
            st.write('Nội dung chính:', games_df.Plots[12][:200],'...')
            row_data = games_df.loc[12, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Alien Isolation"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[12]
                plots= games_df.Plots.values[12]
                
                # DF query
                matches = similarity_df[games_df.Title[12]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[12], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
                    
                    
            st.title(games_df.Title[13])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/6/64/Alliance_Alive_cover_art.jpg/220px-Alliance_Alive_cover_art.jpg")
            st.write('Nội dung chính:', games_df.Plots[13][:200],'...')
            row_data = games_df.loc[13, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Alliance Alive"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[13]
                plots= games_df.Plots.values[13]
                
                # DF query
                matches = similarity_df[games_df.Title[13]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[13], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            st.title(games_df.Title[14])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/6/62/Amnesia-The-Dark-Descent-Cover-Art.png/220px-Amnesia-The-Dark-Descent-Cover-Art.png")
            st.write('Nội dung chính:', games_df.Plots[14][:200],'...')
            row_data = games_df.loc[14, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Amnesia the dark descent"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[14]
                plots= games_df.Plots.values[14]
                
                # DF query
                matches = similarity_df[games_df.Title[14]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[14], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            st.title(games_df.Title[15])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/7/77/Ancestors_Legacy_Steam_Banner.jpg/220px-Ancestors_Legacy_Steam_Banner.jpg")
            st.write('Nội dung chính:', games_df.Plots[15][:200],'...')
            row_data = games_df.loc[15, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Ancestor Legacy"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[15]
                plots= games_df.Plots.values[15]
                
                # DF query
                matches = similarity_df[games_df.Title[15]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[15], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            st.title(games_df.Title[16])
            st.image("https://upload.wikimedia.org/wikipedia/vi/1/1f/Animal_Crossing_New_Horizons.jpg")
            st.write('Nội dung chính:', games_df.Plots[16][:200],'...')
            row_data = games_df.loc[16, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Animal Crossing New horizon"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[16]
                plots= games_df.Plots.values[16]
                
                # DF query
                matches = similarity_df[games_df.Title[16]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[16], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            st.title(games_df.Title[17])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5f/Anonymous%3BCode.png/220px-Anonymous%3BCode.png")
            st.write('Nội dung chính:', games_df.Plots[17][:200],'...')
            row_data = games_df.loc[17, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Anonymous Code"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[17]
                plots= games_df.Plots.values[17]
                
                # DF query
                matches = similarity_df[games_df.Title[17]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[17], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            st.title(games_df.Title[18])
            st.image("https://m.media-amazon.com/images/I/81M7OXORxiL._SY500_.jpg")
            st.write('Nội dung chính:', games_df.Plots[18][:200],'...')
            row_data = games_df.loc[18, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Another world"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[18]
                plots= games_df.Plots.values[18]
                
                # DF query
                matches = similarity_df[games_df.Title[18]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[18], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            st.title(games_df.Title[19])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/d4/Ao-tennis-2-cover.jpg/220px-Ao-tennis-2-cover.jpg")
            st.write('Nội dung chính:', games_df.Plots[19][:200],'...')
            row_data = games_df.loc[19, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Ao tennis 2"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[19]
                plots= games_df.Plots.values[19]
                
                # DF query
                matches = similarity_df[games_df.Title[19]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[19], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
            st.title(games_df.Title[20])
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/9/93/Ape_Out_poster.jpg/220px-Ape_Out_poster.jpg")
            st.write('Nội dung chính:', games_df.Plots[20][:200],'...')
            row_data = games_df.loc[20, cols]
            st.table(pd.DataFrame(row_data).T)
            if st.button("Chọn Ape Out"):
                link = 'https://en.wikipedia.org' + games_df.Link.values[20]
                plots= games_df.Plots.values[20]
                
                # DF query
                matches = similarity_df[games_df.Title[20]].sort_values()[1:6]
                matches = matches.index.tolist()
                matches = games_df.set_index('Title').loc[matches]
                matches.reset_index(inplace=True)

                # Results
                cols = ['Genre', 'Developer', 'Publisher', 'Released in: Japan', 'North America', 'Rest of countries']

                st.markdown("# Những gợi ý cho game  [{}]({}): \n {} ".format(games_df.Title[20], link, plots))
                for idx, row in matches.iterrows():
                    st.markdown('### {} - {}'.format(str(idx + 1), row['Title']))

                    st.markdown(
                        '{} [[...]](https://en.wikipedia.org{})'.format(textwrap.wrap(row['Plots'][0:], 1000)[0], row['Link']))
                    st.table(pd.DataFrame(row[cols]).T.set_index('Genre'))
                    st.markdown('Link to wiki page: [{}](https://en.wikipedia.org{})'.format(row['Title'], row['Link']))
            
            
        
