# 🌫️ Clustering de la Calidad del Aire en Lima Metropolitana

Aplicación web que permite visualizar el agrupamiento de zonas según la calidad del aire en Lima Metropolitana, utilizando algoritmos de clustering sobre datos abiertos del SENAMHI.

> 📍 [Aplicación desplegada en Render](https://clusteringcalidadaire.onrender.com/)

---

## 📊 ¿Qué hace esta app?

- Procesa datos históricos de calidad del aire (PM10, NO2, O3, etc.)
- Aplica algoritmos de clustering (como K-Means) para agrupar zonas
- Visualiza los resultados en mapas y gráficos
- Permite interpretar zonas críticas y patrones de contaminación

---

## 📌 Tecnologías utilizadas

- **Python** + **Flask** 🐍 – Backend ligero y eficiente
- **HTML/CSS/JavaScript** 🖥️ – Interfaz simple y funcional
- **Pandas, Scikit-learn, Matplotlib** 📊 – Análisis y visualización de datos
- **Render.com** 🚀 – Plataforma de despliegue en la nube

---
---

## ⚙️ Instrucciones para Ejecución Local

Puedes ejecutar la aplicación en tu computadora local siguiendo estos pasos:

```bash
1. Clonar el repositorio
git clone https://github.com/tu-usuario/yehinnerta-clusteringcalidadaire.git
cd yehinnerta-clusteringcalidadaire

2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

3. Instalar dependencias
pip install -r requirements.txt

4. Ejecutar
python app.py
