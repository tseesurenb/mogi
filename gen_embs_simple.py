#%%
import openai
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from llm import llm, embeddings

# Load your data
#taxpayer_df = pd.read_csv('confirmed_sellers.csv')
inv_df = pd.read_csv('data/inventory_itstore.csv')

# Data preprocessing
def preprocess_data(df):
    df['received_stock_date'] = pd.to_datetime(df['received_stock_date'])
    df['info_downloaded_date'] = pd.to_datetime(df['info_downloaded_date'])
    df['full_description'] = df.apply(lambda row: f"{row.main_category} {row.category} {row.sub_category} {row.product_name}", axis=1)
    return df

inv_df = preprocess_data(inv_df)

print(inv_df.head())

# %%
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# Enable tqdm for pandas apply
tqdm.pandas(desc="Generating Embeddings")

# Split data into 3 batches
num_batches = 3
batch_size = len(inv_df) // num_batches

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size if i < num_batches - 1 else len(inv_df)
    batch_df = inv_df.iloc[start_idx:end_idx]

    # Apply embeddings with progress bar for each batch
    batch_df['ada_embedding'] = batch_df['full_description'].progress_apply(
        lambda x: get_embedding(x, model='text-embedding-3-small')
    )

    # Save each batch to a separate CSV file
    batch_df.to_csv(f'data/inventory_with_embs_batch_{i+1}.csv', index=False)

# Combine all batches if desired
combined_df = pd.concat([pd.read_csv(f'data/inventory_with_embs_batch_{i+1}.csv') for i in range(num_batches)], ignore_index=True)
combined_df.to_csv('data/inventory_with_embs_full.csv', index=False)

# %%
