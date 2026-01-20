import pandas as pd
from sklearn.preprocessing import StandardScaler

def apply_standardization(raw_df):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(raw_df)
    scaled_df = pd.DataFrame(scaled_df)
    scaled_df.columns = raw_df.columns
    return scaled_df

def extract_seasons_stats(clusters, predictions, real_observations, feature_t, feature_e):
    statistics = []
    for curr_cluster in clusters.itertuples():
        cluster_name = curr_cluster.Index
        cluster_weight = curr_cluster.weight
        cluster = np.where(predictions == cluster_name)[0]
        mean_t, std_t = (
            real_observations.iloc[cluster][feature_t].mean(),
            real_observations.iloc[cluster][feature_t].std(),
        )
        mean_e, std_e = (
            real_observations.iloc[cluster][feature_e].mean(),
            real_observations.iloc[cluster][feature_e].std(),
        )
        statistics.append(
            {
                "cluster_name": cluster_name,
                "freq_rel": cluster_weight,
                "mean_tourists": mean_t,
                "std_tourists": std_t,
                "mean_excursionists": mean_e,
                "std_excursionists": std_e,
            }
        )
    sorted_stat = sorted(statistics, key=lambda x: x["freq_rel"])
    for ind_cluster, clust in enumerate(sorted_stat):
        clust["freq_rel_order"] = ind_cluster
    return sorted_stat