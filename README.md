# Clustering de la Calidad del Aire en Lima Metropolitana

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-web%20framework-black?logo=flask)](https://flask.palletsprojects.com/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6-yellow?logo=javascript)](https://developer.mozilla.org/docs/Web/JavaScript)
[![HTML](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)](https://developer.mozilla.org/docs/Web/HTML)
[![CSS](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)](https://developer.mozilla.org/docs/Web/CSS)
[![Render](https://img.shields.io/badge/Render-Deployed-5D3FD3?logo=render)](https://render.com/)

Aplicación web que permite visualizar el agrupamiento de zonas según la calidad del aire en Lima Metropolitana, utilizando algoritmos de clustering sobre datos abiertos del SENAMHI.

> 📍 [Aplicación desplegada](https://clusteringcalidadaire.onrender.com/)

---

## 📊 ¿Qué hace esta app?

- Procesa datos históricos de calidad del aire (PM10, NO2, O3, etc.)
- Aplica algoritmos de clustering (como K-Means) para agrupar zonas
- Visualiza los resultados en mapas y gráficos
- Permite interpretar zonas críticas y patrones de contaminación

---

## ⚙️ Instrucciones para Ejecución Local

Puedes ejecutar la aplicación en tu computadora local siguiendo estos pasos:

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/yehinnerta-clusteringcalidadaire.git
cd yehinnerta-clusteringcalidadaire

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
python app.py
