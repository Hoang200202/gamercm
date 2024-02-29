
import requests
from bs4 import BeautifulSoup

elements = []
#a access the web
website_url = requests.get('https://en.wikipedia.org/wiki/List_of_PlayStation_games_(A%E2%80%93L)').text

soup = BeautifulSoup(website_url,'lxml')
#print(soup.prettify())

# find table
table = soup.find('table', class_='wikitable plainrowheaders sortable')


for row in table.find_all('tr'):
 

    # only find game names with links to their own pages
    try:
        game = row.find_all('th')
        game_name = game[0].find(text=True)
        game_link = game[0].find(href=True)['href']
    
    except: pass

    # find rest of cells
    else:
        cells = row.find_all('td')
        atts = []
        for i in range(len(cells)-1):
            att = cells[i].find(text=True)
            atts.append(att)
        
        # append full row to list    
        elements.append([game_name, game_link] + atts)

print(elements)

# ====================================================================================================================================

import pandas as pd

cols = ['Title', 'Link', 'Genre', 'Developer', 'Publisher', 'Release_JP']

df_games = pd.DataFrame(elements, columns=cols)
df_games = df_games.astype(str)

# clean links 
df_games = df_games.applymap(lambda x: x.replace('\n', ''))
df_games = df_games.applymap(lambda x: x.replace(':', ''))


print('Shape of dataframe: ', df_games.shape, '\n')
df_games.head() 

# ====================================================================================================================================
# Visit game own pages and extract plots

wiki_url = 'https://en.wikipedia.org'

plots = []
for idx, row in df_games.iterrows():
    
    url = wiki_url + row['Link']
    
    website_url = requests.get(url).text

    soup = BeautifulSoup(website_url,'lxml')
    #print(soup.prettify())

    text = ''
    
    for section in soup.find_all('h2'):
        
        if section.text.startswith('Game') or section.text.startswith('Plot'):

            text += section.text + '\n\n'

            for element in section.next_siblings:
                if element.name and element.name.startswith('h'):
                    break

                elif element.name == 'p':
                    text += element.text + '\n'

        else: pass
    
    if not text:
        plots.append(None)
    else:
        plots.append(text)

# ====================================================================================================================================        
# Clean texts
import re

plots_clean = []
for text in plots:
    if text is not None:
        text = re.sub(r'\[.*?\]+', '', text)
        text = text.replace('\n', ' ')
        text = text.replace('Gameplay ', '')
        text = text.replace('Game-play ', '')
        text = text.replace('Plot ', '')
        plots_clean.append(text)
    else:
        plots_clean.append(None)

df_games['Plots'] = plots_clean

df_games.head()   
     
# ====================================================================================================================================              
# Drop 'Untitled' games
idx_todrop = df_games[df_games.Title=='Untitled '].index.tolist()
df_games.drop(index=idx_todrop, inplace=True)

# Rename cols
rename = {'Release_JP': 'Released in: Japan', 'Release_NA': 'North America', 
 'Release_Pal': 'Rest of countries'}
df_games.rename(columns=rename, inplace=True)        
        
        
df_games.dropna().to_csv('datasets/Games_dataset.csv')        
        
        
