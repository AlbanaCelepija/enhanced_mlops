import os
from pickle import dump
from collections import Counter
from sklearn.mixture import GaussianMixture
from utils import apply_standardization, extract_seasons_stats
from library.src.artifact_types import Data, Configuration, Report, Model


FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACTS_PATH = os.path.join(FOLDER_PATH, "artifacts", "model")


################################################################## Feature Engineering

def permutation_feature_importance(data_test: Data, model: Model):
    # Function to permute a specific feature column in the dataset
    def permute_X(X, j):
        Xj = X.copy()
        Xj[j] = Xj[j].sample(frac=1).values  # Shuffle feature values
        return Xj

    X_test = data_test.get_dataset()
    # Define parameters
    np.random.seed(10)
    n_features = len(X_test.columns)
    n_iter = 10

    # Initialize arrays to store accuracy and disparate impact results
    accs = np.zeros((n_iter, n_features))
    dis = np.zeros((n_iter, n_features))

    # Iterate over features and perform permutation testing
    for j in tqdm(range(n_features)):
        for i in range(n_iter):
            # Shuffle feature j and make predictions
            X_test_permuted = permute_X(X_test, str(j))
            y_pred_test_permuted = model.predict(X_test_permuted)

            # Compute accuracy and disparate impact and store results
            accs[i, j] = accuracy_score(y_test, y_pred_test_permuted)
            dis[i, j] = disparate_impact(
                group_a_test, group_b_test, y_pred_test_permuted
            )


################################################################# Model Training 

def fit_gaussian_mixture_model(raw_df: Data, config: Configuration, nr_clusters=4): 
    scaled_df = apply_standardization(raw_df.get_dataset())
    gmm = GaussianMixture(
        n_components=config.nr_clusters, warm_start=True, covariance_type="diag", max_iter=200
    ).fit(scaled_df)
    gmm_predict = gmm.predict(scaled_df)
    means = gmm.means_
    cluster_predictions = Counter(gmm_predict)
    centers = pd.DataFrame(means, columns=raw_df.columns)
    centers["size"] = [cluster_predictions[i] for i in range(config.nr_clusters)]
    centers["weight"] = centers["size"] / centers["size"].sum()
    statistics = extract_seasons_stats(
        centers, gmm_predict, raw_df, "presences_tourists", "presences_excursionists"
    )
    return statistics   

def model_evaluation_clustering(data: Data, model: Model, config: Configuration):
    pass