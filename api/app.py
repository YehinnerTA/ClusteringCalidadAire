import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non‑interactive backend for Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns  # Optional aesthetics

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # CORS handling

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap  # umap‑learn package
from scipy.cluster.hierarchy import dendrogram, linkage  # For dendrograms

# -----------------------------------------------------------------------------
# Flask app initialisation
# -----------------------------------------------------------------------------
# When this file lives inside the `api/` folder (recommended for Vercel),
# we need to reference the real paths of the "static" and "templates" folders
# that live one level up, next to the project root.
#
#     ├─ api/
#     │   └─ app.py   <-- this file
#     ├─ static/
#     └─ templates/
#
# The lines below calculate those absolute paths so the app works both
# locally and once deployed as a serverless function on Vercel.
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # …/project/api
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))  # …/project

app = Flask(
    __name__,
    static_folder=os.path.join(ROOT_DIR, "static"),
    template_folder=os.path.join(ROOT_DIR, "templates"),
)
CORS(app)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def fig_to_base64(fig):
    """Convert a Matplotlib figure to a base64‑encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# Global list of numerical features expected in every uploaded file
FEATURE_COLUMNS = ["PM10", "PM2_5", "NO2"]

# -----------------------------------------------------------------------------
# Route: /analyze – main entry point used by the front‑end to upload a file and
# receive the clustering results.
# -----------------------------------------------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze_data():
    if "fileUpload" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["fileUpload"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    algorithm = request.form.get("algorithmSelect")
    if not algorithm:
        return jsonify({"error": "No algorithm selected"}), 400

    try:
        # ------------------------------------------------------------------
        # 1. Read Excel file (openpyxl engine if .xlsx)
        # ------------------------------------------------------------------
        df = pd.read_excel(
            file,
            engine="openpyxl" if file.filename.lower().endswith(".xlsx") else None,
        )

        # Check required columns are present
        missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            return (
                jsonify(
                    {
                        "error": f"Missing columns: {', '.join(missing_cols)}. Expected: {', '.join(FEATURE_COLUMNS)}"
                    }
                ),
                400,
            )

        # ------------------------------------------------------------------
        # 2. Basic preprocessing – select columns, drop NaNs
        # ------------------------------------------------------------------
        data = df[FEATURE_COLUMNS].copy().dropna()
        if data.empty:
            return (
                jsonify(
                    {
                        "error": "No data left after dropping rows with NaNs in the selected columns."
                    }
                ),
                400,
            )
        if data.shape[0] < 2:
            return (
                jsonify({"error": "Need at least 2 rows to run clustering."}),
                400,
            )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        X_original = data  # Keep unscaled copy for tree explainability

        results = {"metrics": {}, "plots": {}}
        clusters = None

        # ------------------------------------------------------------------
        # 3. Dimensionality reduction (PCA) for plotting convenience
        # ------------------------------------------------------------------
        n_pca_components = min(2, X_scaled.shape[1], X_scaled.shape[0])
        if n_pca_components < 1:
            return jsonify({"error": "Not enough data for PCA"}), 400
        try:
            pca_result = PCA(n_components=n_pca_components, random_state=42).fit_transform(
                X_scaled
            )
        except ValueError as e_pca:
            return (
                jsonify(
                    {
                        "error": f"PCA failed – perhaps the data is constant or singular. Details: {str(e_pca)}"
                    }
                ),
                400,
            )

        # ------------------------------------------------------------------
        # 4. Choose clustering algorithm
        # ------------------------------------------------------------------
        if algorithm == "kmeans":
            # -------------------- K‑Means --------------------
            max_k = min(10, X_scaled.shape[0] - 1) or 1
            sse = []
            for k in range(1, max_k + 1):
                kmeans_tmp = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X_scaled)
                sse.append(kmeans_tmp.inertia_)
            # Elbow plot
            fig_elbow, ax_elbow = plt.subplots(figsize=(8, 5))
            ax_elbow.plot(range(1, max_k + 1), sse, marker="o")
            ax_elbow.set_xticks(range(1, max_k + 1))
            ax_elbow.set_xlabel("Número de Clusters (k)")
            ax_elbow.set_ylabel("SSE")
            ax_elbow.set_title("Método del Codo")
            ax_elbow.grid(True)
            results["plots"]["elbow_method"] = fig_to_base64(fig_elbow)

            optimal_k = min(3, X_scaled.shape[0]) or 1
            kmeans = KMeans(n_clusters=optimal_k, n_init="auto", random_state=42).fit(
                X_scaled
            )
            clusters = kmeans.labels_
            if len(np.unique(clusters)) > 1:
                results["metrics"].update(
                    {
                        "silhouette": f"{silhouette_score(X_scaled, clusters):.3f}",
                        "calinski_harabasz": f"{calinski_harabasz_score(X_scaled, clusters):.3f}",
                        "davies_bouldin": f"{davies_bouldin_score(X_scaled, clusters):.3f}",
                    }
                )
            else:
                results["metrics"]["silhouette"] = "No calculable (1 cluster)"
            results["metrics"]["num_clusters"] = optimal_k

        elif algorithm == "dbscan":
            # -------------------- DBSCAN --------------------
            eps_val = 0.8
            min_samples_val = min(10, X_scaled.shape[0] - 1) or 1
            dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val).fit(X_scaled)
            clusters = dbscan.labels_
            n_clusters_db = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise_db = list(clusters).count(-1)
            results["metrics"].update(
                {
                    "num_clusters_dbscan": n_clusters_db,
                    "num_noise_points_dbscan": n_noise_db,
                }
            )
            if n_clusters_db > 1:
                results["metrics"]["silhouette_dbscan"] = f"{silhouette_score(X_scaled, clusters):.3f}"
            else:
                results["metrics"]["silhouette_dbscan"] = "No calculable / mucho ruido"

            # DBSCAN PCA scatter
            fig_db_pca, ax_db_pca = plt.subplots(figsize=(8, 6))
            scatter = ax_db_pca.scatter(
                pca_result[:, 0], pca_result[:, 1], c=clusters, cmap="Dark2", s=10, alpha=0.7
            )
            ax_db_pca.set_title("DBSCAN sobre PCA")
            ax_db_pca.set_xlabel("PC1")
            ax_db_pca.set_ylabel("PC2")
            ax_db_pca.grid(True)
            results["plots"]["dbscan_pca"] = fig_to_base64(fig_db_pca)

        elif algorithm == "hierarchical":
            # -------------------- Agglomerative Clustering --------------------
            n_clusters_h = min(3, X_scaled.shape[0]) or 1
            ag = AgglomerativeClustering(n_clusters=n_clusters_h).fit(X_scaled)
            clusters = ag.labels_
            if len(np.unique(clusters)) > 1:
                results["metrics"].update(
                    {
                        "silhouette_hierarchical": f"{silhouette_score(X_scaled, clusters):.3f}",
                        "calinski_harabasz_hierarchical": f"{calinski_harabasz_score(X_scaled, clusters):.3f}",
                        "davies_bouldin_hierarchical": f"{davies_bouldin_score(X_scaled, clusters):.3f}",
                    }
                )
            else:
                results["metrics"]["silhouette_hierarchical"] = "No calculable (1 cluster)"
            results["metrics"]["num_clusters_hierarchical"] = n_clusters_h

            # Dendrogram (small datasets only)
            if 1 < X_scaled.shape[0] < 1000:
                linked = linkage(X_scaled, method="ward")
                fig_dendro, ax_dendro = plt.subplots(figsize=(12, 7))
                dendrogram(linked, orientation="top", distance_sort="descending", truncate_mode="lastp", p=12, ax=ax_dendro)
                ax_dendro.set_title("Dendrograma Jerárquico")
                ax_dendro.set_xticks([])
                results["plots"]["dendrogram"] = fig_to_base64(fig_dendro)
        else:
            return jsonify({"error": "Algoritmo no implementado"}), 400

        # ------------------------------------------------------------------
        # 5. Common visualisations (Decision tree, PCA, t‑SNE, UMAP)
        # ------------------------------------------------------------------
        if clusters is not None:
            # Decision tree (skip if only noise / single cluster)
            non_noise_mask = clusters != -1 if -1 in clusters else np.ones_like(clusters, dtype=bool)
            valid_clusters = clusters[non_noise_mask]
            if len(np.unique(valid_clusters)) > 1:
                clf = DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_leaf=5)
                clf.fit(X_original[non_noise_mask], valid_clusters)
                fig_tree, ax_tree = plt.subplots(figsize=(18, 10))
                plot_tree(
                    clf,
                    feature_names=X_original.columns.tolist(),
                    class_names=[str(c) for c in sorted(np.unique(valid_clusters))],
                    filled=True,
                    rounded=True,
                    fontsize=10,
                    ax=ax_tree,
                    impurity=False,
                    proportion=True,
                )
                ax_tree.set_title(f"Árbol de Decisión – {algorithm.upper()}")
                results["plots"]["decision_tree"] = fig_to_base64(fig_tree)

            # PCA scatter (works for any cluster count ≥1)
            fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
            scatter_pca = ax_pca.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap="Set1", s=15, alpha=0.7)
            ax_pca.set_title(f"Clusters ({algorithm.upper()}) – PCA")
            ax_pca.set_xlabel("PC1")
            ax_pca.set_ylabel("PC2")
            ax_pca.grid(True)
            results["plots"]["pca_clusters"] = fig_to_base64(fig_pca)

            # t‑SNE (skip on huge datasets)
            if 1 < X_scaled.shape[0] < 5000:
                perplexity_val = min(30, X_scaled.shape[0] - 1)
                tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, n_iter=300, init="pca")
                tsne_res = tsne.fit_transform(X_scaled)
                fig_tsne, ax_tsne = plt.subplots(figsize=(8, 6))
                scatter_tsne = ax_tsne.scatter(tsne_res[:, 0], tsne_res[:, 1], c=clusters, cmap="Set1", s=15, alpha=0.7)
                ax_tsne.set_title(f"Clusters ({algorithm.upper()}) – t‑SNE")
                ax_tsne.set_xlabel("t‑SNE1")
                ax_tsne.set_ylabel("t‑SNE2")
                ax_tsne.grid(True)
                results["plots"]["tsne_clusters"] = fig_to_base64(fig_tsne)
            else:
                results["plots"]["tsne_clusters_message"] = "t‑SNE omitido: dataset muy grande o muy pequeño."

            # UMAP
            if X_scaled.shape[0] > 1:
                n_neighbors_val = min(15, X_scaled.shape[0] - 1)
                reducer = umap.UMAP(n_neighbors=n_neighbors_val, min_dist=0.1, n_components=2, random_state=42)
                umap_res = reducer.fit_transform(X_scaled)
                fig_umap, ax_umap = plt.subplots(figsize=(8, 6))
                scatter_umap = ax_umap.scatter(umap_res[:, 0], umap_res[:, 1], c=clusters, cmap="Set1", s=15, alpha=0.7)
                ax_umap.set_title(f"Clusters ({algorithm.upper()}) – UMAP")
                ax_umap.set_xlabel("UMAP1")
                ax_umap.set_ylabel("UMAP2")
                ax_umap.grid(True)
                results["plots"]["umap_clusters"] = fig_to_base64(fig_umap)
            else:
                results["plots"]["umap_clusters_message"] = "UMAP omitido: datos insuficientes."

        if clusters is None and "message" not in results:
            results["message"] = "El algoritmo no produjo clusters válidos."

        return jsonify(results)

    # ----------------------------------------------------------------------
    # Error handling
    # ----------------------------------------------------------------------
    except MemoryError as e_mem:
        return (
            jsonify({"error": f"Error de memoria: {str(e_mem)}"}),
            500,
        )
    except ValueError as e_val:
        return (
            jsonify({"error": f"Error de valor: {str(e_val)}"}),
            400,
        )
    except Exception as e:
        return (
            jsonify({"error": f"Ocurrió un error inesperado: {str(e)}"}),
            500,
        )

# -----------------------------------------------------------------------------
# Front page – renders templates/index.html
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -----------------------------------------------------------------------------
# Local development runner (ignored by Vercel – it just imports `app`)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure directories exist when running locally
    os.makedirs(os.path.join(ROOT_DIR, "templates"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "static"), exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
