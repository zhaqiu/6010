import matplotlib.pyplot as plt
import pandas as pd
#plot user data by age and by gender
df = pd.read_csv('C:/Users/user/PycharmProjects/HKUST5140/data/members.csv', encoding='utf8')
age = []
user_ids = []
gender = []
for index, row in df.iterrows():
    if row['bd']==0 or row['bd']<5 or row['bd']>95 :
        continue
    age.append(row['bd'])
    user_ids.append(row['msno'])
    gender.append(row['gender'])

user_age_data = pd.DataFrame({'user_id': user_ids, 'age':age})
hist = user_age_data['age'].plot.kde()
print(user_age_data['age'].mean())
x_value = []
for i in range(0,100):
    if(i%5==0):
        x_value.append(i)
plt.xticks(x_value)
hist.set_xlim(0, 100)
plt.title('user age distribution')
plt.show()
new_df = df.groupby(by='gender').size()
new_df.plot.pie(autopct='%1.0f%%')
plt.title('user by gender')
plt.show()


#plot song length distribution
df = pd.read_csv('C:/Users/user/PycharmProjects/HKUST5140/data/songs.csv', encoding='utf8')
print( df['song_length'].mean()/(1000*60))
song_length_seconds = []
for index, row in df.iterrows():
    song_length_seconds.append(row['song_length']/(1000*60))
df['song_length_seconds'] = song_length_seconds
hist = df['song_length_seconds'].plot.kde()
plt.xticks([0, 1, 2, 3,4,5,6, 7 , 8 , 9])
hist.set_xlim(0, 9)
plt.xlabel('song length in minutes', fontsize=18)
plt.show()
df = pd.read_csv('C:/Users/user/PycharmProjects/HKUST5140/data/train_data.csv', encoding='utf8')
for index, row in df.iterrows():
    x = row['user_id']

count_row = df.shape[0]

df['count'] = [1]*count_row
new_df = df.groupby(by='target').size()
#new_df = df.groupby(by='msno')['count'].sum()
print(new_df)
new_df.plot.bar()
plt.title('training data count by target value')
plt.show()