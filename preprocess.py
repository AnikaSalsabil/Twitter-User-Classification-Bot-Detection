import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read the dataset
df = pd.read_csv('twitter_user_data.csv' , encoding='latin1')
df.head()


#2 types of unit_state : finalized and golden 
df._unit_state.unique()

#2 types of unit_state : finalized and golden 
df._trusted_judgments.unique()


#by goruping the dtaaset by trusted judgements in relation to unit_state we see that golden unit states has trusted judgemnts above 200 and all other got around 3 . SO goldens are reliable and less contaminated. 
'''Unit_state golds can be used as test sets'''
df._trusted_judgments.groupby(df._unit_state).count()
df[df['_unit_state'] == 'golden']['_trusted_judgments'].values 



#13926 observations has confidence level of 1 on gender and rest of them are less than 1.
'''following rows Can be used for training classification model for gender identification the obeservations with confidence==1 and was judged by contributors.  '''
df[ (df['profile_yn:confidence'] == 1) & (df['profile_yn'] =='yes')].shape[0]


#created oclumn : NOt important 


#description
df['description'] 
'''can be tranformed '''


#fav_number
'''Drop column because no real value'''





#shows us 20000  rows have null value for this column 
'''delete bcz of insuffucuent data
It only appears for a tiny subset of rows (just 46 in your dataset), so it’s sparse and not representative'''
df['gender_gold'].isnull().sum()
print(df.gender_gold.value_counts())
print(df[(df['gender_gold'] == 'male') |( df['gender_gold'] == 'female' )| (df['gender_gold'] == 'brand' )  ].shape)







#profile_yn_gold
'''Only the golden 50 rows has 'yes' value rest of them null  '''
df['profile_yn_gold'].value_counts() #which gives 50 yes and rest null
df[(df['_unit_state'] == 'golden') ]['profile_yn_gold'].shape #shows us all 50 obserbations has gold _uni_state so we can say: 
'''We candelete the column because we can rely on _unit_state column and use it for same purpose as the column and due to its accessive null values it better ot delete it to avoid bias'''



#retweet_count
print(df[(df['retweet_count'] == 1) & (df['gender'] =='brand')].shape)
print(df[(df['retweet_count'] == 1) & (df['gender'] =='male')].shape)
print(df[(df['retweet_count'] == 1) & (df['gender'] =='female')].shape)
'''We can say that the column retweet_count is not very useful for our purpose, because the values observed for each entity, such as 149, 184, or 143, appear arbitrary and carry no real significance in establishing a relationship for differentiating entities.'''





#   DROPPING COLUMNS

'''decided to drop these columns because they primarily consist of identifiers, cosmetic profile settings, or simple activity counts, which in my judgment do not contribute meaningful insights for distinguishing between human and non-human profiles within the scope of this text- and attribute-based analysis.'''
df.drop(columns=['_unit_id' , 'fav_number' , 'link_color' , 'name' , 'profileimage' , 'sidebar_color' , 'tweet_coord' , 'tweet_count' , 'created' , 'tweet_id' , 'tweet_location' , 'user_timezone' , 'profile_yn_gold' , 'retweet_count' , '_last_judgment_at' , 'gender_gold' , 'tweet_created'] , axis=1 , inplace=True)



#  HANDLING NULLS

'''having no description is itself a signal (e.g., many bots or brands may not write descriptions), we can keep those rows and replace nulls with a placeholder like "missing_description"'''

#see how null is represented
df[df['description'].isnull()].head()

df.description.fillna('missing_description' , inplace=True)



'''Delete insignificant 82 rows with missing gender  values  and change  and 

Dropping all rows with gender marked as "unknown" because they do not provide a clear label. 
Keeping them would introduce noise into the dataset and reduce model accuracy, 
while removing them ensures the classifier learns only from reliable categories (male, female, brand).

'''
df.dropna(subset=['gender'] , inplace=True)
df = df[df['gender'] != "unknown"]



'''Because all the profiles has "yes" value for profile_yn we can remove that column too '''
df['profile_yn'].value_counts()
df.drop(columns=['profile_yn'] , axis=1 , inplace=True)



'''Want to keep only the profile that has confidence in their observation and lates as all the profile has full confidence  the column has no meaning'''
df['profile_yn:confidence'].value_counts()
df = df[df['profile_yn:confidence'] >= 1 ]
df.drop(columns=['profile_yn:confidence'] , inplace=True , axis=1)

'''in all rows,  all of the False _goldens are just "finalized" and True _goldens are "golden" so these 2 columns are eventually same we can keep only one  '''
df.groupby(['_golden', '_unit_state']).size()
df.drop(columns=['_unit_state'] , axis=1 , inplace=True)


#create out directory if not exists



import os 
out_dir = os.path.expanduser("./out")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#save cleaned data
df.to_csv("./out/df_cleaned.csv", index=False)