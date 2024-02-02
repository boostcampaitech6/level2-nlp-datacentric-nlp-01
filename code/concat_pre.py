import pandas as pd

file_paths = [f'../data/augmentation_{i}.csv' for i in range(7)]

dfs = [pd.read_csv(path, index_col=0) for path in file_paths]

con_df = pd.concat(dfs, ignore_index=True)
con_df.rename(columns={'label': 'target'}, inplace=True)

con_df['text'] = con_df['text'].str.replace(r'[^\w\s%.]', '', regex=True)

con_df['text'] = con_df['text'].str.replace(r'\.{2,}', '.', regex=True)
        
cleaned_data = pd.read_csv('../data/cleaned_data.csv')


result = pd.concat([con_df, cleaned_data[['text', 'target']]], axis=0, ignore_index=True)
result = result[result['text'].apply(lambda x: isinstance(x, str))]
#print(result.head())

result.to_csv('before_cleanlab_train.csv',index=False)
