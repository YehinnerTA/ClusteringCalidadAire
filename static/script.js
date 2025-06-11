document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('uploadForm');
    const loadingDiv = document.getElementById('loading');
    const errorMessageDiv = document.getElementById('error-message');
    const resultsArea = document.getElementById('resultsArea');
    const metricsContent = document.getElementById('metricsContent');

    // Textos de explicación para cada gráfico
    const plotExplanationTexts = {
        elbow: "El Método del Codo ayuda a seleccionar el número óptimo de clusters (k) para K-Means. Se busca un 'codo' en la gráfica, que representa el punto donde añadir más clusters ya no reduce significativamente la suma de los errores cuadráticos (SSE) dentro de los clusters.",
        dbscanPca: "Esta gráfica muestra los clusters identificados por DBSCAN proyectados en los dos primeros Componentes Principales (PCA). DBSCAN agrupa puntos que están densamente conectados y marca como 'ruido' los puntos aislados. Los colores distinguen los clusters y el ruido.",
        pca: "Visualización de los clusters utilizando Análisis de Componentes Principales (PCA). PCA reduce la dimensionalidad de los datos a dos componentes principales, intentando retener la mayor varianza posible. Cada color representa un cluster diferente.",
        tsne: "Visualización de los clusters utilizando t-SNE (t-distributed Stochastic Neighbor Embedding). t-SNE es una técnica de reducción de dimensionalidad no lineal, especialmente buena para visualizar la estructura de datos de alta dimensionalidad en un espacio de baja dimensión (2D o 3D).",
        umap: "Visualización de los clusters utilizando UMAP (Uniform Manifold Approximation and Projection). UMAP es otra técnica moderna de reducción de dimensionalidad que a menudo preserva bien tanto la estructura global como local de los datos.",
        dendrogram: "El Dendrograma es una visualización jerárquica de los clusters. Muestra cómo los puntos de datos individuales o pequeños clusters se fusionan progresivamente en clusters más grandes. La altura de las uniones indica la (dis)imilitud entre los clusters.",
        decisionTree: "Este Árbol de Decisión intenta explicar cómo se forman los clusters. Cada nodo representa una decisión basada en los valores de los contaminantes (PM10, PM2.5, NO2), y las hojas indican la asignación a un cluster. Ayuda a interpretar qué características definen cada grupo."
    };

    const plotElements = {
        elbow: { 
            container: document.getElementById('plotElbow'), 
            img: document.getElementById('imgElbow'),
            explanationP: document.getElementById('explanationElbow')
        },
        dbscanPca: { 
            container: document.getElementById('plotDbscanPca'), 
            img: document.getElementById('imgDbscanPca'),
            explanationP: document.getElementById('explanationDbscanPca')
        },
        pca: { 
            container: document.getElementById('plotPca'), 
            img: document.getElementById('imgPca'),
            explanationP: document.getElementById('explanationPca')
        },
        tsne: { 
            container: document.getElementById('plotTsne'), 
            img: document.getElementById('imgTsne'), 
            msg: document.getElementById('tsneMessage'),
            explanationP: document.getElementById('explanationTsne')
        },
        umap: { 
            container: document.getElementById('plotUmap'), 
            img: document.getElementById('imgUmap'), 
            msg: document.getElementById('umapMessage'),
            explanationP: document.getElementById('explanationUmap')
        },
        dendrogram: { 
            container: document.getElementById('plotDendrogram'), 
            img: document.getElementById('imgDendrogram'), 
            msg: document.getElementById('dendrogramMessage'),
            explanationP: document.getElementById('explanationDendrogram')
        },
        decisionTree: { 
            container: document.getElementById('plotDecisionTree'), 
            img: document.getElementById('imgDecisionTree'), 
            msg: document.getElementById('decisionTreeMessage'),
            explanationP: document.getElementById('explanationDecisionTree')
        }
    };

    form.addEventListener('submit', async function (event) {
        event.preventDefault();
        
        loadingDiv.style.display = 'flex';
        errorMessageDiv.style.display = 'none';
        resultsArea.style.display = 'none';
        metricsContent.innerHTML = '';
        Object.values(plotElements).forEach(el => {
            if (el.container) el.container.style.display = 'none';
            if (el.img) el.img.src = '#';
            if (el.msg) el.msg.textContent = '';
            if (el.explanationP) el.explanationP.textContent = ''; // Limpiar explicaciones anteriores
        });

        const formData = new FormData(form);
        const selectedAlgorithm = document.getElementById('algorithmSelect').value;

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            loadingDiv.style.display = 'none';

            if (response.ok) {
                resultsArea.style.display = 'block';
                displayMetrics(data.metrics, selectedAlgorithm);
                displayPlots(data.plots, selectedAlgorithm);
                
                if (data.message) {
                    const msgPara = document.createElement('p');
                    msgPara.textContent = data.message;
                    metricsContent.appendChild(msgPara);
                }

            } else {
                errorMessageDiv.textContent = `Error: ${data.error || 'Ocurrió un error desconocido.'}`;
                errorMessageDiv.style.display = 'block';
            }
        } catch (error) {
            loadingDiv.style.display = 'none';
            errorMessageDiv.textContent = `Error de red o servidor: ${error.message}`;
            errorMessageDiv.style.display = 'block';
        }
    });

    function displayMetrics(metrics, algorithm) {
        metricsContent.innerHTML = ''; 
        if (Object.keys(metrics).length === 0 && !metrics.num_clusters_dbscan) { // Considerar DBSCAN que puede no tener todas las métricas
            metricsContent.innerHTML = '<p>No se generaron métricas estándar o el algoritmo no las produce.</p>';
            //return; // No retornar si queremos mostrar métricas específicas de DBSCAN
        }

        let metricsHtml = `<h4>Métricas para ${algorithm.toUpperCase()}</h4>`;
        let hasMetrics = false;
        for (const key in metrics) {
            hasMetrics = true;
            let displayName = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            if (key === "silhouette") displayName = "Silhouette Score (K-Means)";
            if (key === "calinski_harabasz") displayName = "Calinski-Harabasz Score (K-Means)";
            if (key === "davies_bouldin") displayName = "Davies-Bouldin Score (K-Means)";
            if (key === "num_clusters") displayName = "Número de Clusters (K-Means)";
            if (key === "num_clusters_dbscan") displayName = "Número de Clusters (DBSCAN)";
            if (key === "num_noise_points_dbscan") displayName = "Puntos de Ruido (DBSCAN)";
            if (key === "silhouette_dbscan") displayName = "Silhouette Score (DBSCAN)";
            if (key === "silhouette_hierarchical") displayName = "Silhouette Score (Jerárquico)";
            if (key === "calinski_harabasz_hierarchical") displayName = "Calinski-Harabasz Score (Jerárquico)";
            if (key === "davies_bouldin_hierarchical") displayName = "Davies-Bouldin Score (Jerárquico)";
            if (key === "num_clusters_hierarchical") displayName = "Número de Clusters (Jerárquico)";
            metricsHtml += `<p><strong>${displayName}:</strong> ${metrics[key]}</p>`;
        }
         if (!hasMetrics) {
            metricsContent.innerHTML = '<p>No se generaron métricas para este algoritmo/resultado.</p>';
        } else {
            metricsContent.innerHTML = metricsHtml;
        }
    }

    function displayPlots(plots, algorithm) {
        Object.values(plotElements).forEach(el => {
            if(el.container) el.container.style.display = 'none';
            if(el.explanationP) el.explanationP.textContent = '';
            if(el.msg) {
                el.msg.textContent = '';
                el.msg.style.display = 'none';
            }
            if(el.img && el.img.style) el.img.style.display = 'block';
        });

        function showPlot(plotKey, plotData) {
            const element = plotElements[plotKey];
            if (element && element.img && element.container) {
                element.img.src = `data:image/png;base64,${plotData}`;
                element.container.style.display = 'block';
                element.img.style.display = 'block'; // Asegurar que la imagen sea visible
                if (element.explanationP && plotExplanationTexts[plotKey]) {
                    element.explanationP.textContent = plotExplanationTexts[plotKey];
                }
            }
        }
        
        function showPlotMessage(plotKey, messageKey) {
            const element = plotElements[plotKey];
             if (element && element.msg && plots[messageKey]) {
                element.msg.textContent = plots[messageKey];
                element.msg.style.display = 'block';
                if (element.container) element.container.style.display = 'block';
                if (element.img) element.img.style.display = 'none';
                if (element.explanationP && plotExplanationTexts[plotKey]) { // También mostrar explicación si hay mensaje
                    element.explanationP.textContent = plotExplanationTexts[plotKey] + " (Nota: Gráfico no generado - ver mensaje arriba)";
                }
            }
        }

        if (plots.elbow_method && algorithm === 'kmeans') {
            showPlot('elbow', plots.elbow_method);
        }
        if (plots.dbscan_pca && algorithm === 'dbscan') {
            showPlot('dbscanPca', plots.dbscan_pca);
        }
        if (plots.pca_clusters) { // Este es común a todos los algoritmos que producen clusters
            showPlot('pca', plots.pca_clusters);
        }
        
        if (plots.tsne_clusters) {
            showPlot('tsne', plots.tsne_clusters);
        } else if (plots.tsne_clusters_message) {
            showPlotMessage('tsne', 'tsne_clusters_message');
        }
        
        if (plots.umap_clusters) {
            showPlot('umap', plots.umap_clusters);
        } else if (plots.umap_clusters_message) {
            showPlotMessage('umap', 'umap_clusters_message');
        }
        
        if (plots.dendrogram && algorithm === 'hierarchical') {
            showPlot('dendrogram', plots.dendrogram);
        } else if (plots.dendrogram_message && algorithm === 'hierarchical') {
            showPlotMessage('dendrogram', 'dendrogram_message');
        }
        
        if (plots.decision_tree) {
            showPlot('decisionTree', plots.decision_tree);
        } else if (plots.decision_tree_message) {
            showPlotMessage('decisionTree', 'decision_tree_message');
        }
    }
});