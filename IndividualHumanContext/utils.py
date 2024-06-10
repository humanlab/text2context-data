from tqdm import tqdm
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def get_embeddings(df, column, model_name, layer=-2):
    """
    Get embeddings for a given column in a dataframe
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)
    model.eval()
    
    BATCH_SIZE = 16 if device == torch.device("cuda") else 1
    
    msgs = df[column].values.tolist()
    embeddings = []
    
    for idx in tqdm(range(0, len(msgs), BATCH_SIZE)):
        batch_msgs = msgs[idx:idx+BATCH_SIZE]
        tokens = tokenizer(batch_msgs, return_tensors="pt", padding=True, truncation=True)
        tokens = {k:v.to(device) for k, v in tokens.items()}
        model_output = model(**tokens, output_hidden_states=True)
        model_output = model_output.hidden_states[layer]
        model_output = (tokens['attention_mask'].unsqueeze(-1)* model_output).sum(dim=1) 
        model_output = model_output / tokens['attention_mask'].sum(dim=1).unsqueeze(-1)
        model_output = model_output.detach().cpu().numpy().tolist()
        embeddings.extend(model_output)
        
    return np.array(embeddings)


def continuous_factor_adaptation(language_features:np.ndarray, user_factors:np.ndarray):
    """
    Adapt the language features to the user factors
    """
    # For each user factor, create a copy of language feature by multiplying it with the user factor
    adapted_features = []
    adapted_features.append(language_features)
    for user_factor_idx in range(user_factors.shape[1]):
        user_factor = user_factors[:, user_factor_idx, None]
        adapted_features.append(language_features * user_factor)
    
    # Now hstack the adapted features
    adapted_features = np.hstack(adapted_features)
    return adapted_features


def zscore(data:np.ndarray, mean:np.ndarray=None, std:np.ndarray=None):
    """
    Z-score normalization
    """
    if mean is None: mean = np.mean(data, axis=0)
    if std is None: std = np.std(data, axis=0)
    return ((data - mean) / std), mean, std


def eval_performance(y_true:np.ndarray, y_pred:np.ndarray):
    """
    Evaluate the performance of the model. Compute Pearson correlation between y_true and y_pred
    """
    return pearsonr(y_true, y_pred)[0]


def eval_performance_user_level(y_true:np.ndarray, y_pred:np.ndarray, user_ids: np.ndarray):
    """
    Evaluate the performance of the model. Compute Pearson correlation for each user, and then average them.
    If a user has less than 2 samples, we ignore that user.
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'user_id': user_ids})
    corr_values = []
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id]
        if len(user_data) < 2:
            continue
        corr_values.append(pearsonr(user_data['y_true'], user_data['y_pred'])[0])
    return np.mean(corr_values) if len(corr_values) > 0 else 0.0
    