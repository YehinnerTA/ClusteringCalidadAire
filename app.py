import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns # For better aesthetics on some plots, optional

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # To handle Cross-Origin Resource Sharing

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap # umap-learn package
from scipy.cluster.hierarchy import dendrogram, linkage # For dendrogram

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Helper Function to Generate Plots ---
def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded string."""
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.read()).decode('utf-8')

# --- Global variable for selected features ---
FEATURE_COLUMNS = ['PM10', 'PM2_5', 'NO2']

# --- Main Analysis Route ---
@app.route('/analyze', methods=['POST'])
def analyze_data():
    if 'fileUpload' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['fileUpload']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    algorithm = request.form.get('algorithmSelect')
    if not algorithm:
        return jsonify({"error": "No algorithm selected"}), 400

    try:
        # Read Excel file
        # Consider using openpyxl explicitly if needed for .xlsx
        df = pd.read_excel(file, engine='openpyxl' if file.filename.endswith('.xlsx') else None)
        
        # --- Data Preprocessing ---
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns in uploaded file: {', '.join(missing_cols)}. Expected: {', '.join(FEATURE_COLUMNS)}"}), 400

        # Use only the specified feature columns
        data_for_clustering = df[FEATURE_COLUMNS].copy()
        data_for_clustering.dropna(inplace=True) 

        if data_for_clustering.empty:
            return jsonify({"error": "No data remaining after preprocessing (after dropping NaNs from selected columns). Check your data."}), 400
        
        if data_for_clustering.shape[0] < 2: # Need at least 2 samples for most operations
             return jsonify({"error": "Not enough data points (less than 2) after preprocessing."}), 400

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_clustering)
        
        X_for_tree = data_for_clustering # Original (unscaled) data for tree interpretability

        results = {"metrics": {}, "plots": {}}
        clusters = None 

        # --- Dimensionality Reduction for Visualization (do this once) ---
        # PCA for general visualization
        # Ensure n_components is not more than n_features or n_samples
        n_pca_components = min(2, scaled_data.shape[1], scaled_data.shape[0])
        if n_pca_components < 1: # Should not happen if shape[0] check above is done
            return jsonify({"error": "Not enough features/samples for PCA."}), 400
        
        pca_2d = PCA(n_components=n_pca_components, random_state=42)
        try:
            pca_result = pca_2d.fit_transform(scaled_data)
        except ValueError as e_pca_val:
             return jsonify({"error": f"PCA failed. Data might be unsuitable (e.g., all zeros after scaling). Error: {str(e_pca_val)}"}), 400


        # --- Algorithm Specific Logic ---
        if algorithm == 'kmeans':
            # 1. Elbow Method
            sse = []
            # Ensure K_range does not exceed number of samples
            max_k_elbow = min(10, scaled_data.shape[0] -1 if scaled_data.shape[0] > 1 else 1)
            if max_k_elbow < 1: max_k_elbow = 1 # Handle edge case of 1 sample
            K_range = range(1, max_k_elbow + 1)
            
            if len(K_range) > 0:
                for k_val in K_range:
                    kmeans_elbow = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                    kmeans_elbow.fit(scaled_data)
                    sse.append(kmeans_elbow.inertia_)
                
                fig_elbow, ax_elbow = plt.subplots(figsize=(8, 5))
                ax_elbow.plot(K_range, sse, marker='o')
                if len(K_range) > 0: ax_elbow.set_xticks(K_range)
                ax_elbow.set_xlabel("Número de Clusters (k)")
                ax_elbow.set_ylabel("Error Cuadrático Total (SSE)")
                ax_elbow.set_title("Método del Codo para Determinar k Óptimo")
                ax_elbow.grid(True)
                results["plots"]["elbow_method"] = fig_to_base64(fig_elbow)
            else:
                results["plots"]["elbow_method_message"] = "No se pudo generar el método del codo (muy pocos datos)."


            # 2. K-Means Clustering
            # Ensure optimal_k does not exceed number of samples
            optimal_k = min(3, scaled_data.shape[0]) # Default or determined from elbow
            if optimal_k < 1: optimal_k = 1

            if optimal_k > 0 :
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
                clusters = kmeans.fit_predict(scaled_data)
                
                if len(np.unique(clusters)) > 1: # silhouette needs at least 2 clusters
                    results["metrics"]["silhouette"] = f"{silhouette_score(scaled_data, clusters):.3f}"
                    results["metrics"]["calinski_harabasz"] = f"{calinski_harabasz_score(scaled_data, clusters):.3f}"
                    results["metrics"]["davies_bouldin"] = f"{davies_bouldin_score(scaled_data, clusters):.3f}"
                else:
                    results["metrics"]["silhouette"] = "No calculable (solo 1 cluster)"
                results["metrics"]["num_clusters"] = optimal_k
            else:
                 results["message"] = "No se pudo ejecutar K-Means (muy pocos datos para el k seleccionado)."


        elif algorithm == 'dbscan':
            eps_val = 0.8 
            min_samples_val = min(10, scaled_data.shape[0] -1 if scaled_data.shape[0] > 0 else 1)
            if min_samples_val < 1: min_samples_val = 1

            dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
            clusters = dbscan.fit_predict(scaled_data)
            
            n_clusters_db = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise_db = list(clusters).count(-1)
            results["metrics"]["num_clusters_dbscan"] = n_clusters_db
            results["metrics"]["num_noise_points_dbscan"] = n_noise_db
            
            if n_clusters_db > 1:
                results["metrics"]["silhouette_dbscan"] = f"{silhouette_score(scaled_data, clusters):.3f}"
            elif n_clusters_db == 1 and n_noise_db == 0 and len(set(clusters)) == 1: # Only one cluster, no noise
                 results["metrics"]["silhouette_dbscan"] = "No calculable (solo 1 cluster sin ruido)."
            else: # Mix of one cluster and noise, or all noise
                results["metrics"]["silhouette_dbscan"] = "DBSCAN solo detectó un cluster o ruido."
            
            # DBSCAN on PCA plot (only if PCA result exists)
            if 'pca_result' in locals() and pca_result.shape[1] >=2:
                fig_db_pca, ax_db_pca = plt.subplots(figsize=(8, 6))
                scatter = ax_db_pca.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='Dark2', s=10, alpha=0.7)
                ax_db_pca.set_title(f"DBSCAN (eps={eps_val}, min_samples={min_samples_val}) sobre PCA")
                ax_db_pca.set_xlabel("Componente Principal 1")
                ax_db_pca.set_ylabel("Componente Principal 2")
                ax_db_pca.grid(True)
                unique_labels_db = np.unique(clusters)
                handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {l}' if l != -1 else 'Ruido',
                                      markerfacecolor=scatter.cmap(scatter.norm(l))) for l in unique_labels_db]
                ax_db_pca.legend(handles=handles, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
                results["plots"]["dbscan_pca"] = fig_to_base64(fig_db_pca)

        elif algorithm == 'hierarchical':
            n_clusters_hierarchical = min(3, scaled_data.shape[0]) 
            if n_clusters_hierarchical < 1: n_clusters_hierarchical = 1

            if n_clusters_hierarchical > 0 and scaled_data.shape[0] >= n_clusters_hierarchical :
                agg_clustering = AgglomerativeClustering(n_clusters=n_clusters_hierarchical)
                clusters = agg_clustering.fit_predict(scaled_data)
                
                if len(np.unique(clusters)) > 1:
                    results["metrics"]["silhouette_hierarchical"] = f"{silhouette_score(scaled_data, clusters):.3f}"
                    results["metrics"]["calinski_harabasz_hierarchical"] = f"{calinski_harabasz_score(scaled_data, clusters):.3f}"
                    results["metrics"]["davies_bouldin_hierarchical"] = f"{davies_bouldin_score(scaled_data, clusters):.3f}"
                else:
                    results["metrics"]["silhouette_hierarchical"] = "No calculable (solo 1 cluster)"
                results["metrics"]["num_clusters_hierarchical"] = n_clusters_hierarchical
                
                # Dendrogram (can be complex for large datasets)
                if scaled_data.shape[0] > 1 and scaled_data.shape[0] < 1000: # Limit for performance/readability
                    try:
                        linked = linkage(scaled_data, method='ward')
                        fig_dendro, ax_dendro = plt.subplots(figsize=(12, 7))
                        dendrogram(linked,
                                   orientation='top',
                                   distance_sort='descending',
                                   truncate_mode='lastp', # Show only the last p merged clusters
                                   p=12, # Number of merged clusters to show
                                   show_leaf_counts=False, # Avoids clutter
                                   ax=ax_dendro)
                        ax_dendro.set_title('Dendrograma Jerárquico')
                        plt.xticks([]) # Hide x-axis labels for leaf nodes
                        results["plots"]["dendrogram"] = fig_to_base64(fig_dendro)
                    except Exception as e_dendro:
                        results["plots"]["dendrogram_message"] = f"No se pudo generar el dendrograma: {str(e_dendro)}"
                        print(f"Error dendrogram: {e_dendro}")
                elif scaled_data.shape[0] >= 1000:
                     results["plots"]["dendrogram_message"] = "Dendrograma omitido (dataset demasiado grande)."

            else:
                 results["message"] = "No se pudo ejecutar Clustering Jerárquico (muy pocos datos)."
        else:
            return jsonify({"error": "Algoritmo no implementado"}), 400

        # --- Common Visualizations and Decision Tree Explanation ---
        if clusters is not None:
            # Ensure pca_result exists and has 2 components for plotting
            can_plot_pca = 'pca_result' in locals() and pca_result.shape[1] >= 2

            # 1. Decision Tree for Explaining Clusters
            unique_cluster_labels = np.unique(clusters)
            # Need at least 2 distinct non-noise clusters for a meaningful tree
            mask_not_noise = (clusters != -1) if -1 in unique_cluster_labels else np.ones(len(clusters), dtype=bool)
            valid_clusters_for_tree = clusters[mask_not_noise]
            
            if np.sum(mask_not_noise) > 0 and len(np.unique(valid_clusters_for_tree)) > 1:
                tree_explainer = DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_leaf=5)
                tree_explainer.fit(X_for_tree[mask_not_noise], valid_clusters_for_tree)
                
                fig_tree, ax_tree = plt.subplots(figsize=(18, 10)) # Increased size
                plot_tree(tree_explainer, 
                          feature_names=X_for_tree.columns.tolist(), 
                          class_names=[str(i) for i in sorted(np.unique(valid_clusters_for_tree))], 
                          filled=True, 
                          rounded=True,
                          fontsize=10, # Adjusted fontsize
                          ax=ax_tree,
                          impurity=False, # Cleaner look
                          proportion=True) # Show proportions
                ax_tree.set_title(f"Árbol de Decisión Explicando los Clusters ({algorithm.upper()})", fontsize=16)
                results["plots"]["decision_tree"] = fig_to_base64(fig_tree)
            else:
                results["plots"]["decision_tree_message"] = "No hay suficientes clusters (excluyendo ruido) o datos para generar el árbol de decisión."

            # 2. PCA Plot of Clusters
            if can_plot_pca:
                fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
                scatter_pca = ax_pca.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='Set1', s=15, alpha=0.7)
                ax_pca.set_title(f"Clusters ({algorithm.upper()}) por PCA")
                ax_pca.set_xlabel("Componente Principal 1")
                ax_pca.set_ylabel("Componente Principal 2")
                ax_pca.grid(True)
                if len(np.unique(clusters)) > 0: # Check if there are any clusters to make a legend for
                     handles_pca = [plt.Line2D([0], [0], marker='o', color='w', 
                                               label=f'Cluster {l}' if l!=-1 else 'Ruido',
                                               markerfacecolor=scatter_pca.cmap(scatter_pca.norm(l))) 
                                    for l in np.unique(clusters)]
                     ax_pca.legend(handles=handles_pca, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
                results["plots"]["pca_clusters"] = fig_to_base64(fig_pca)

            # 3. t-SNE Plot
            if scaled_data.shape[0] > 1 and scaled_data.shape[0] < 5000: # Limit for performance
                # Perplexity must be less than n_samples
                perplexity_val = min(30, scaled_data.shape[0] - 1)
                if perplexity_val > 0:
                    try:
                        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_iter=300, init='pca') # init='pca' for stability
                        tsne_result = tsne.fit_transform(scaled_data)
                        fig_tsne, ax_tsne = plt.subplots(figsize=(8, 6))
                        scatter_tsne = ax_tsne.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters, cmap='Set1', s=15, alpha=0.7)
                        ax_tsne.set_title(f"Clusters ({algorithm.upper()}) por t-SNE")
                        ax_tsne.set_xlabel("t-SNE Componente 1")
                        ax_tsne.set_ylabel("t-SNE Componente 2")
                        ax_tsne.grid(True)
                        if len(np.unique(clusters)) > 0:
                            handles_tsne = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {l}' if l!=-1 else 'Ruido',
                                                     markerfacecolor=scatter_tsne.cmap(scatter_tsne.norm(l))) for l in np.unique(clusters)]
                            ax_tsne.legend(handles=handles_tsne, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
                        results["plots"]["tsne_clusters"] = fig_to_base64(fig_tsne)
                    except Exception as e_tsne:
                        results["plots"]["tsne_clusters_message"] = f"t-SNE no pudo generarse: {str(e_tsne)}"
                        print(f"Error en t-SNE: {e_tsne}")
                else:
                    results["plots"]["tsne_clusters_message"] = "t-SNE omitido (muy pocos datos para perplexity)."
            elif scaled_data.shape[0] >= 5000 :
                results["plots"]["tsne_clusters_message"] = "t-SNE omitido debido al gran tamaño del dataset."
            else: # scaled_data.shape[0] <= 1
                 results["plots"]["tsne_clusters_message"] = "t-SNE omitido (datos insuficientes)."


            # 4. UMAP Plot
            if scaled_data.shape[0] > 1: # UMAP needs at least 2 samples
                # n_neighbors must be less than n_samples
                n_neighbors_val = min(15, scaled_data.shape[0] - 1)
                if n_neighbors_val > 0:
                    try:
                        reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors_val, min_dist=0.1, n_components=2)
                        umap_result = reducer.fit_transform(scaled_data)
                        fig_umap, ax_umap = plt.subplots(figsize=(8, 6))
                        scatter_umap = ax_umap.scatter(umap_result[:, 0], umap_result[:, 1], c=clusters, cmap='Set1', s=15, alpha=0.7)
                        ax_umap.set_title(f"Clusters ({algorithm.upper()}) por UMAP")
                        ax_umap.set_xlabel("UMAP Componente 1")
                        ax_umap.set_ylabel("UMAP Componente 2")
                        ax_umap.grid(True)
                        if len(np.unique(clusters)) > 0:
                            handles_umap = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {l}' if l!=-1 else 'Ruido',
                                                     markerfacecolor=scatter_umap.cmap(scatter_umap.norm(l))) for l in np.unique(clusters)]
                            ax_umap.legend(handles=handles_umap, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
                        results["plots"]["umap_clusters"] = fig_to_base64(fig_umap)
                    except Exception as e_umap:
                         results["plots"]["umap_clusters_message"] = f"UMAP no pudo generarse: {str(e_umap)}"
                         print(f"Error en UMAP: {e_umap}")
                else:
                    results["plots"]["umap_clusters_message"] = "UMAP omitido (muy pocos datos para n_neighbors)."
            else: # scaled_data.shape[0] <= 1
                 results["plots"]["umap_clusters_message"] = "UMAP omitido (datos insuficientes)."


        elif clusters is None and not results.get("message"): # No clusters and no specific message yet
            results["message"] = "El algoritmo no produjo resultados de clustering."
        elif not results.get("message"): # clusters is not None but some other condition made it skip common plots
            # Check if only one cluster was found and provide a message
            if clusters is not None and len(np.unique(clusters)) == 1:
                results["message"] = "El algoritmo de clustering encontró solo un cluster. Algunas visualizaciones de comparación de clusters pueden no ser aplicables."
                # Still, try to plot the single cluster on PCA if possible
                if can_plot_pca:
                    fig_pca_single, ax_pca_single = plt.subplots(figsize=(8, 6))
                    ax_pca_single.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='Set1', s=15, alpha=0.7)
                    ax_pca_single.set_title(f"Datos (Un Solo Cluster por {algorithm.upper()}) por PCA")
                    ax_pca_single.set_xlabel("Componente Principal 1")
                    ax_pca_single.set_ylabel("Componente Principal 2")
                    ax_pca_single.grid(True)
                    results["plots"]["pca_clusters"] = fig_to_base64(fig_pca_single)

        return jsonify(results)

    except MemoryError as e_mem:
        print(f"MemoryError during analysis: {e_mem}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error de Memoria: El dataset es demasiado grande para el algoritmo '{algorithm}' con los recursos actuales. Prueba con un archivo más pequeño, un muestreo de datos, o un algoritmo más escalable como K-Means. Error: {str(e_mem)}"}), 500
    except ValueError as e_val: # Catch other ValueErrors that might not be specific
        print(f"ValueError during analysis: {e_val}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error de Valor: Puede haber un problema con los datos o los parámetros. Verifica tu archivo. Error: {str(e_val)}"}), 400
    except Exception as e:
        print(f"Error during analysis: {e}") 
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Ocurrió un error inesperado durante el análisis: {str(e)}"}), 500

@app.route('/')
def index():
    return render_template('index.html') 

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)