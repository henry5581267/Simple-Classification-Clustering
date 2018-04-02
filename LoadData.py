import pandas as pd
import numpy as np
# =============================================================================
#                            CPBL Player Data
# =============================================================================
# Read CSV
df_CPBL = pd.read_csv('CPBL1317_UTF8.csv', sep=',')
# Get column
df_CPBL_year = df_CPBL.loc[:, 'Year']
df_CPBL_goaboad = df_CPBL.loc[:, 'Goabroad']
df2_CPBL = df_CPBL.loc[:, 'PA':'SLG']
df_CPBL_team = df_CPBL.loc[:, 'Team']


#%% Normalization
minix_CPBL = np.array([df2_CPBL['PA'].min(), df2_CPBL['AVG'].min(
), df2_CPBL['OBP'].min(), df2_CPBL['SLG'].min()])
maxix_CPBL = np.array([df2_CPBL['PA'].max(), df2_CPBL['AVG'].max(
), df2_CPBL['OBP'].max(), df2_CPBL['SLG'].max()])


df2_CPBL['PA'] = (df2_CPBL['PA']-df2_CPBL['PA'].min()) / \
    (df2_CPBL['PA'].max()-df2_CPBL['PA'].min())

df2_CPBL['AVG'] = (df2_CPBL['AVG']-df2_CPBL['AVG'].min()) / \
    (df2_CPBL['AVG'].max()-df2_CPBL['AVG'].min())

df2_CPBL['OBP'] = (df2_CPBL['OBP']-df2_CPBL['OBP'].min()) / \
    (df2_CPBL['OBP'].max()-df2_CPBL['OBP'].min())

df2_CPBL['SLG'] = (df2_CPBL['SLG']-df2_CPBL['SLG'].min()) / \
    (df2_CPBL['SLG'].max()-df2_CPBL['SLG'].min())

# Onehot Encoding
# Year
onehot_encoding_CPBL_year = pd.get_dummies(df_CPBL_year, prefix='Year')
# Once Goabroad or not
onehot_encoding_goabroad = pd.get_dummies(df_CPBL_goaboad, prefix='Goabroad')
df2_CPBL = pd.concat(
    [onehot_encoding_CPBL_year, onehot_encoding_goabroad, df2_CPBL, ], axis=1)

# Team
# Team mapping
onehot_encoding_CPBL_team = pd.get_dummies(df_CPBL_team, prefix='Team')
#%%
# =============================================================================
#                           Car Evaluation Data Set
# =============================================================================
df_car = pd.read_csv('Car.csv', sep=',', header=None,)
# Process data
df_car.columns = ['buying', 'maint', 'doors',
                  'persons', 'lug_boot', 'safety', 'Class']
df2_car = df_car.loc[:, 'buying':'safety']

buying_mapping = {
    'vhigh': 4,
    'high': 3,
    'med': 2,
    'low': 1
}

maint_mapping = buying_mapping
doors_mapping = {
    '2': 1,
    '3': 2,
    '4': 3,
    '5more': 4
}
persons_mapping = {
    '2': 1,
    '4': 2,
    'more': 3
}
lug_boot_mapping = {
    'small': 1,
    'med': 2,
    'big': 3
}
safety_mapping = {
    'low': 1,
    'med': 2,
    'high': 3
}
df2_car['buying'] = df_car.replace({'buying': buying_mapping})
df2_car['maint'] = df_car['maint'].map(maint_mapping)
df2_car['doors'] = df_car['doors'].map(doors_mapping)
df2_car['persons'] = df_car['persons'].map(persons_mapping)
df2_car['lug_boot'] = df_car['lug_boot'].map(lug_boot_mapping)
df2_car['safety'] = df_car['safety'].map(safety_mapping)
onehot_encoding_car = pd.get_dummies(df_car['Class'], prefix='Class')
# Normalization
df2_car['buying'] = (df2_car['buying']-df2_car['buying'].min()) / \
    (df2_car['buying'].max()-df2_car['buying'].min())
df2_car['maint'] = (df2_car['maint']-df2_car['maint'].min()) / \
    (df2_car['maint'].max()-df2_car['maint'].min())
df2_car['doors'] = (df2_car['doors']-df2_car['doors'].min()) / \
    (df2_car['doors'].max()-df2_car['doors'].min())
df2_car['persons'] = (df2_car['persons']-df2_car['persons'].min()) / \
    (df2_car['persons'].max()-df2_car['persons'].min())
df2_car['lug_boot'] = (df2_car['lug_boot']-df2_car['lug_boot'].min()) / \
    (df2_car['lug_boot'].max()-df2_car['lug_boot'].min())
df2_car['safety'] = (df2_car['safety']-df2_car['safety'].min()) / \
    (df2_car['safety'].max()-df2_car['safety'].min())


# =============================================================================
# Abalone Data Set
# =============================================================================
#df=pd.read_csv('Abalone.csv', sep=',',header=None)
#df.columns = ['sex','length','diameter','height','whole weight','shucked weight','viscera weight','shell weight','rings']
#df2 = df.loc[:,'length':'shell weight']
#onehot_encoding = pd.get_dummies(df['sex'],prefix = 'sex')
# Normalization
#df2['length'] = (df2['length']-df2['length'].min())/(df2['length'].max()-df2['length'].min())
#df2['diameter'] = (df2['diameter']-df2['diameter'].min())/(df2['diameter'].max()-df2['diameter'].min())
#df2['height'] = (df2['height']-df2['height'].min())/(df2['height'].max()-df2['height'].min())
#df2['whole weight'] = (df2['whole weight']-df2['whole weight'].min())/(df2['whole weight'].max()-df2['whole weight'].min())
#df2['shucked weight'] = (df2['shucked weight']-df2['shucked weight'].min())/(df2['shucked weight'].max()-df2['shucked weight'].min())
#df2['viscera weight'] = (df2['viscera weight']-df2['viscera weight'].min())/(df2['viscera weight'].max()-df2['viscera weight'].min())
#df2['shell weight'] = (df2['shell weight']-df2['shell weight'].min())/(df2['shell weight'].max()-df2['shell weight'].min())
#df2 = pd.concat([onehot_encoding,df2],axis =1)
#mini = df['rings'].min()
#maxi =  df['rings'].max()
#df['rings'] = (df['rings'] - df['rings'].min())/(df['rings'].max()-df['rings'].min())

# =============================================================================
#                               Training Data
# =============================================================================
# Car
x_car_data = df2_car.values
y_car_data = onehot_encoding_car.values

# Abalone Data Set
#x_data = df2.values
#y_data = df['rings'].values

# CPBL Player
x_CPBL_data = df2_CPBL.values
y_CPBL_data = onehot_encoding_CPBL_team.values
