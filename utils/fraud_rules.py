# utils/fraud_rules.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime

# Liste de pays "à risque" d'exemple (tu peux adapter)
HIGH_RISK_COUNTRIES = {"NG", "RU", "UA", "PK", "BD", "IR"}

def rule_score(tx):
    """
    tx : dict ou Series contenant keys: amount, country, hour, user_mean_amount, tx_count_last_24h
    Retourne score (0..100) et motifs list
    """
    score = 0
    reasons = []

    amount = float(tx.get("amount", 0))
    country = str(tx.get("country", "")).upper()
    hour = int(tx.get("hour", 12))
    user_mean = float(tx.get("user_mean_amount", 0))
    tx24 = int(tx.get("tx_count_last_24h", 0))

    # règle 1 : montant très supérieur à la moyenne utilisateur
    if user_mean > 0 and amount > 5 * user_mean:
        score += 30
        reasons.append("Montant > 5× moyenne utilisateur")

    # règle 2 : montant élevé absolu
    if amount > 2000:
        score += 25
        reasons.append("Montant élevé (> 2000)")

    # règle 3 : pays à risque
    if country in HIGH_RISK_COUNTRIES:
        score += 20
        reasons.append(f"Pays à risque ({country})")

    # règle 4 : transaction en dehors des heures usuelles (ex: 0-5h)
    if hour < 6 or hour > 22:
        score += 10
        reasons.append("Horaire inhabituel")

    # règle 5 : activité très élevée sur 24h
    if tx24 > 10:
        score += 15
        reasons.append("Beaucoup de transactions en 24h")

    # clamp
    score = min(100, score)
    return score, reasons

def enrich_transactions(df):
    """
    Ajoute colonnes utiles : hour, user_mean_amount, tx_count_last_24h
    df expected columns: transaction_id, user_id, amount, timestamp, country
    """
    df = df.copy()
    # parse timestamp to hour
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour

    # mean amount per user
    df['user_mean_amount'] = df.groupby('user_id')['amount'].transform('mean')

    # count per user last 24h: for simplicity, compute count in same day
    df['tx_count_last_24h'] = df.groupby(['user_id', df['timestamp'].dt.date])['transaction_id'].transform('count')

    # compute rule score and reasons
    scores = []
    reasons = []
    for _, row in df.iterrows():
        s, r = rule_score(row)
        scores.append(s)
        reasons.append("; ".join(r) if r else "")
    df['rule_score'] = scores
    df['rule_reasons'] = reasons

    return df

def ml_anomaly_score(df, features=None, random_state=42):
    """
    IsolationForest to detect anomalies: returns anomaly_score (1 = normal, -1 = anomaly)
    and anomaly_rate (float score).
    """
    df = df.copy()
    if features is None:
        features = ['amount', 'hour', 'user_mean_amount', 'tx_count_last_24h']

    X = df[features].fillna(0).values
    iso = IsolationForest(contamination=0.03, random_state=random_state)
    iso.fit(X)
    pred = iso.predict(X)            # 1 normal, -1 anomaly
    score = -iso.decision_function(X)  # higher = more anomalous (negate so larger = worse)

    df['ml_pred'] = pred
    df['ml_score_raw'] = score
    # normalize ml_score_raw to 0..100
    minv, maxv = score.min(), score.max()
    if maxv - minv == 0:
        df['ml_score'] = 0
    else:
        df['ml_score'] = ((score - minv) / (maxv - minv) * 100).clip(0,100)

    return df

def combined_score(df, weight_rule=0.6, weight_ml=0.4):
    """
    Combine rule_score and ml_score into final score (0..100)
    """
    df = df.copy()
    df['final_score'] = (df['rule_score'] * weight_rule + df['ml_score'] * weight_ml).clip(0,100)
    # risk label
    def label(s):
        if s < 30:
            return "Faible"
        elif s < 60:
            return "Moyen"
        else:
            return "Élevé"
    df['risk_label'] = df['final_score'].apply(label)
    return df
