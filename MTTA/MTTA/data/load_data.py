import kagglehub
import os

path = kagglehub.dataset_download("qingyi/wm811k-wafer-map")

print("REAL dataset path:", path)
print("Files:", os.listdir(path))