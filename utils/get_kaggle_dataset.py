import kagglehub
import os

os.environ["KAGGLEHUB_CACHE"] = "./data/"

try:
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
except:
    print("Download failed, attempting via cURL")
    os.system(
        "curl -L -o ~/Downloads/brazilian-ecommerce.zip\
  https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
    )

print("Path to dataset files:", path)
