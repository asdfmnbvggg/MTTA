import pandas as pd
from sklearn.model_selection import train_test_split

pkl_path = r"C:\Users\1423\Downloads\MTTA\MTTA-2\MTTA\MTTA\data\LSWD_id.pkl"

train_path = r"C:\Users\1423\Downloads\MTTA\MTTA-2\MTTA\MTTA\data\LSWD_id_train.pkl"
test_path  = r"C:\Users\1423\Downloads\MTTA\MTTA-2\MTTA\MTTA\data\LSWD_id_test.pkl"

df = pd.read_pickle(pkl_path)

print("전체 데이터 수:", len(df))

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Train:", len(train_df))
print("Test :", len(test_df))

train_df.to_pickle(train_path)
test_df.to_pickle(test_path)

print("저장 완료!")
