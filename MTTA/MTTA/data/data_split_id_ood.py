import pandas as pd

PKL_PATH = r"C:\Users\1423\Downloads\MTTA\-\MTTA\MTTA\data\LSWMD_prepro.pkl"
df = pd.read_pickle(PKL_PATH)

ood_classes = ['Scratch', 'Random', 'Donut', 'Near-full']

df_ood = df[df['failureType_norm'].isin(ood_classes)].copy()
df_id  = df[~df['failureType_norm'].isin(ood_classes)].copy()

df_ood.to_pickle("LSWD_ood.pkl")
df_id.to_pickle("LSWD_id.pkl")

print("OOD class counts:")
print(df_ood['failureType_norm'].value_counts())

print("\nID class counts:")
print(df_id['failureType_norm'].value_counts())

print("\nSaved files:")
print(" - LSWD_ood.pkl")
print(" - LSWD_id.pkl")