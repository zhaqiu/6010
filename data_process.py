import numpy as np
import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import hashlib

folder = 'C:/Users/user/PycharmProjects/HKUST5140/data/'

#
# Prepare the data
#

# songs = pd.read_csv(folder+ 'songs.csv', encoding='utf8')
# song_id = []
# song_length = []
# genre_ids = []
# artist_name = []
# composer = []
# lyricist = []
# language = []
# for index, row in songs.iterrows():
#     print(index)
#     song_id.append(int(hashlib.sha256(row['song_id'].encode('utf-8')).hexdigest(), 16) % 10**8)
#     song_length.append(row['song_length'])
#     if isinstance(row['genre_ids'], float ):
#         genre_ids.append(0)
#     else:
#         genre_ids.append(int(hashlib.sha256(row['genre_ids'].encode('utf-8')).hexdigest(), 32) % 10 ** 5)
#     artist_name.append(int(hashlib.sha256(row['artist_name'].encode('utf-8')).hexdigest(), 32) % 10**7)
#
#     if isinstance(row['composer'], float ):
#         composer.append(0)
#     else:
#         composer.append(int(hashlib.sha256(row['composer'].encode('utf-8')).hexdigest(), 32) % 10 ** 7)
#
#     if isinstance(row['lyricist'], float ):
#         lyricist.append(0)
#     else:
#         lyricist.append(int(hashlib.sha256(row['lyricist'].encode('utf-8')).hexdigest(), 32) % 10 ** 7)
#     language.append(row['language'])
#
# songs_new = pd.DataFrame({'song_id': song_id, 'song_length':song_length, 'genre_ids': genre_ids,
#                           'artist_name':artist_name,'composer':composer,'lyricist':lyricist, 'language':language, })
#
# songs_new.to_csv(folder+ 'songs_new.csv', index=False)

# members = pd.read_csv(folder+ 'members.csv')
# user_ids = []
# city = []
# gender = []
# age = []
# for index, row in members.iterrows():
#     print(index)
#     if row['bd']==0 or row['bd']<5 or row['bd']>95 :
#         age.append(0)
#     else:
#         age.append(row['bd'])
#
#     user_ids.append(int(hashlib.sha256(row['msno'].encode('utf-8')).hexdigest(), 16) % 10**9)
#     if row['gender'] == 'female':
#         gender.append(2)
#     elif row['gender'] == 'male':
#         gender.append(1)
#     else:
#         gender.append(0)
#     city.append(row['city'])
#
# members_new = pd.DataFrame({'user_id': user_ids, 'age':age, 'gender': gender,
#                           'city':city, })
#
# members_new.to_csv(folder+ 'members_new.csv', index=False)
#
# train = pd.read_csv(folder+ 'train.csv')
# user_ids = []
# song_ids = []
# source_system_tab = []
# source_screen_name = []
# source_type	= []
# target = []
#
#
# target_value = []
# for index, row in train.iterrows():
#     print(index)
#     user_ids.append(int(hashlib.sha256(row['msno'].encode('utf-8')).hexdigest(), 16) % 10**9);
#     song_ids.append(int(hashlib.sha256(row['song_id'].encode('utf-8')).hexdigest(), 16) % 10**8)
#     if isinstance(row['source_system_tab'], float ):
#         source_system_tab.append(0)
#     else:
#         source_system_tab.append(int(hashlib.sha256(row['source_system_tab'].encode('utf-8')).hexdigest(), 16) % 10 ** 7)
#
#     if row['source_screen_name'] is None or isinstance(row['source_screen_name'], float ):
#         source_screen_name.append(0)
#     else:
#         source_screen_name.append(int(hashlib.sha256(row['source_screen_name'].encode('utf-8')).hexdigest(), 32) % 10**7)
#
#     if isinstance(row['source_type'], float ):
#         source_type.append(0)
#     else:
#         source_type.append(int(hashlib.sha256(row['source_type'].encode('utf-8')).hexdigest(), 16) % 10 ** 7)
#     target_value.append(row['target'])
#
# train_new = pd.DataFrame({'user_id': user_ids, 'song_id':song_ids, 'source_system_tab': source_system_tab,
#                           'source_screen_name':source_screen_name,
#                           'source_type': source_type,
#                           'target': target_value})
#
# train_new.to_csv(folder+ 'train_new.csv', index=False)




















songs = pd.read_csv(folder+ 'songs_new.csv', encoding='utf8')
members = pd.read_csv(folder+ 'members_new.csv', encoding='utf8')
train = pd.read_csv(folder+ 'train_new.csv', encoding='utf8')
user_id = []
source_system_tab = []
source_screen_name = []
source_type	= []
target = []
song_id = []
song_length = []
genre_ids = []
artist_name = []
composer = []
lyricist = []
language = []
city = []
gender = []
age = []
for index, row in train.iterrows():
    print(index)
    song_row = songs.loc[songs['song_id'] == row['song_id']]
    member_row = members.loc[members['user_id'] == row['user_id']]
    if song_row.shape[0] != 1 or member_row.shape[0] != 1:
        continue

    user_id.append(row['user_id'])
    source_system_tab.append(row['source_system_tab'])
    source_screen_name.append(row['source_screen_name'])
    source_type.append(row['source_type'])
    target.append(row['target'])
    song_id.append(row['song_id'])
    song_length.append(song_row['song_length'].iloc[0])
    genre_ids.append(song_row['genre_ids'].iloc[0])
    artist_name.append(song_row['artist_name'].iloc[0])
    composer.append(song_row['composer'].iloc[0])
    lyricist.append(song_row['lyricist'].iloc[0])
    language.append(song_row['language'].iloc[0])
    city.append(member_row['city'].iloc[0])
    gender.append(member_row['gender'].iloc[0])
    age.append(member_row['age'].iloc[0])


train_new = pd.DataFrame({'user_id': user_id,
                          'song_id':song_id,
                          'source_system_tab': source_system_tab,
                          'source_screen_name':source_screen_name,
                          'source_type': source_type,
                          'song_length':song_length,
                          'genre_ids':genre_ids,
                          'artist_name':artist_name,
                          'composer':composer,
                          'lyricist':lyricist,
                          'language':language,
                          'city':city,
                          'gender':gender,
                          'age':age,
                          'target': target})

train_new.to_csv(folder+ 'train_data.csv', index=False)
