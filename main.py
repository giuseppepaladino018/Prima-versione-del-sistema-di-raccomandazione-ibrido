import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# =========================================================
# Configurazioni
# =========================================================
PESO_COLLABORATIVO_DEFAULT = 0.5
TOP_N_DEFAULT = 5
SEED_RANDOM = 42

# =========================================================
# 1. Caricamento dati
# =========================================================
def carica_dataset(path_ratings='ratings.csv', path_movies='movies.csv'):
    if not os.path.exists(path_ratings) or not os.path.exists(path_movies):
        raise FileNotFoundError("File ratings.csv o movies.csv non trovato.")
    
    valutazioni = pd.read_csv(path_ratings)
    film = pd.read_csv(path_movies)
    
    return valutazioni, film

# =========================================================
# 2. Preprocessing
# =========================================================
def costruisci_matrice_utente_film(valutazioni):
    return valutazioni.pivot_table(
        index='userId', 
        columns='movieId', 
        values='rating'
    ).fillna(0)

def estrai_feature_film(film):
    generi = film['genres'].fillna('')
    df_generi = generi.str.get_dummies(sep='|')
    df_generi.index = film['movieId'].values
    return df_generi

def normalizza_feature(feature_df):
    scaler = StandardScaler()
    arr = scaler.fit_transform(feature_df.values)
    return arr, scaler

# =========================================================
# 3. Filtraggio Collaborativo
# =========================================================
def calcola_similarita_utenti(matrice_utente_film):
    sim = cosine_similarity(matrice_utente_film.values)
    return pd.DataFrame(sim, index=matrice_utente_film.index, columns=matrice_utente_film.index)

def raccomanda_collaborativo(id_utente, matrice, sim_utenti, top_n=None):
    if id_utente not in matrice.index:
        raise KeyError(f"Utente {id_utente} non presente.")
    
    punteggi_sim = sim_utenti.loc[id_utente]
    numeratore = matrice.T.dot(punteggi_sim)
    denominatore = punteggi_sim.sum() + 1e-9
    punteggi = numeratore / denominatore
    
    film_gia_visti = matrice.loc[id_utente]
    candidati = punteggi[film_gia_visti == 0]
    
    candidati_sorted = candidati.sort_values(ascending=False)
    return candidati_sorted.head(top_n) if top_n else candidati_sorted

# =========================================================
# 4. Filtraggio Content-Based
# =========================================================
def calcola_similarita_film(feature_scalate, movie_ids):
    sim = cosine_similarity(feature_scalate)
    return pd.DataFrame(sim, index=movie_ids, columns=movie_ids)

def raccomanda_contenuto(id_utente, matrice, sim_film, top_n=None):
    if id_utente not in matrice.index:
        raise KeyError(f"Utente {id_utente} non presente.")
    
    valutazioni = matrice.loc[id_utente]
    film_visti = valutazioni[valutazioni > 0].index.tolist()
    if not film_visti:
        return pd.Series(dtype=float)
    
    punteggi = sim_film[film_visti].sum(axis=1)
    punteggi = punteggi.drop(labels=film_visti, errors='ignore')
    
    candidati_sorted = punteggi.sort_values(ascending=False)
    return candidati_sorted.head(top_n) if top_n else candidati_sorted

# =========================================================
# 5. Fusione Ibrida
# =========================================================
def normalizza_min_max(serie):
    if serie.empty:
        return serie
    mn, mx = serie.min(), serie.max()
    if mx - mn == 0:
        return pd.Series(0.0, index=serie.index)
    return (serie - mn) / (mx - mn + 1e-9)

def raccomanda_ibrido(id_utente, matrice, sim_utenti, sim_film, peso_collaborativo=PESO_COLLABORATIVO_DEFAULT, top_n=TOP_N_DEFAULT):
    cf = raccomanda_collaborativo(id_utente, matrice, sim_utenti, top_n=None)
    cb = raccomanda_contenuto(id_utente, matrice, sim_film, top_n=None)
    
    cf_norm = normalizza_min_max(cf)
    cb_norm = normalizza_min_max(cb)
    
    all_ids = cf_norm.index.union(cb_norm.index)
    cf_aligned = cf_norm.reindex(all_ids).fillna(0.0)
    cb_aligned = cb_norm.reindex(all_ids).fillna(0.0)
    
    punteggi = peso_collaborativo * cf_aligned + (1 - peso_collaborativo) * cb_aligned
    return punteggi.sort_values(ascending=False).head(top_n)

# =========================================================
# 6. Utility
# =========================================================
def stampa_raccomandazioni(id_utente, matrice, sim_utenti, sim_film, film_df, top_n=TOP_N_DEFAULT, peso_collaborativo=PESO_COLLABORATIVO_DEFAULT):
    risultati = raccomanda_ibrido(id_utente, matrice, sim_utenti, sim_film, peso_collaborativo, top_n)
    
    print(f"\nRaccomandazioni per utente {id_utente}:")
    for mid, score in risultati.items():
        titolo = film_df.set_index('movieId').loc[mid]['title']
        print(f"- {titolo} (id={mid}) -> punteggio: {score:.4f}")

# =========================================================
# 7. Main
# =========================================================
def main(argv):
    valutazioni, film = carica_dataset()
    matrice = costruisci_matrice_utente_film(valutazioni)
    features = estrai_feature_film(film)
    features_scalate, _ = normalizza_feature(features)
    
    sim_utenti = calcola_similarita_utenti(matrice)
    sim_film = calcola_similarita_film(features_scalate, features.index.tolist())
    
    id_utente = int(argv[1]) if len(argv) > 1 else 1
    top_n = int(argv[2]) if len(argv) > 2 else TOP_N_DEFAULT
    peso = float(argv[3]) if len(argv) > 3 else PESO_COLLABORATIVO_DEFAULT
    
    stampa_raccomandazioni(id_utente, matrice, sim_utenti, sim_film, film, top_n, peso)

if __name__ == "__main__":
    main(sys.argv)
