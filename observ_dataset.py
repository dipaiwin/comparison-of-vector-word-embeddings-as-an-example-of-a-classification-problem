import pandas as pd

df = pd.read_csv('./first_test/train.csv')
n_df = pd.DataFrame(columns=['titel', 'id'])
bad_index = [3, 7, 9]
i = 0
for item in range(len(df.index)):
    id_text = df['id'].loc[item]
    text = df['titel'].loc[item]
    if id_text not in bad_index:
        n_df.loc[i] = [text, id_text]
        i += 1
n_df.to_csv('./first_test/my_test.csv')
