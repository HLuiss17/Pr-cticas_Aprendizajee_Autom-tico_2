{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "946be36d-17bf-4b50-a36f-743a62f52b39",
   "metadata": {},
   "source": [
    "![ITQ](imgLogo.png)\n",
    "![ITQ](imgPortada.png)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0cd656c4-8605-41a7-92bd-8b32f85780fc",
   "metadata": {},
   "source": [
    "Luis Pilaguano"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "afe814be-3a05-48a7-818c-4e2d55425546",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iniciando proceso de entrenamiento y exportación...\n",
      "Medias de características calculadas. Shape: (30,)\n",
      "Umbral óptimo encontrado: 0 con accuracy: 0.6268\n",
      "\n",
      "--- ¡PROCESO TERMINADO! ---\n",
      "Archivos 'mp_neuron_model.pkl' y 'feature_means.pkl' creados en este directorio.\n",
      "Ahora puede ejecutar 'python app.py'\n"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "import numpy as np\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "# ==========================================================\n",
    "# 1. Definición de la Clase (Copia exacta de tu Notebook)\n",
    "# ==========================================================\n",
    "class MPNeuron:\n",
    "    \"\"\"Clase que implementa la Neurona de McCulloch y Pitts.\"\"\"\n",
    "    def __init__(self, threshold=0):\n",
    "        self.threshold = threshold\n",
    "        \n",
    "    def model(self, x):\n",
    "        \"\"\"Función de activación: 1 si la suma >= umbral, 0 en caso contrario.\"\"\"\n",
    "        # La MPNeuron suma las entradas binarias\n",
    "        return (np.sum(x) >= self.threshold).astype(int)\n",
    "    \n",
    "    def predict(self, X):\n",
    "        \"\"\"Genera predicciones para un conjunto de datos X.\"\"\"\n",
    "        return np.array([self.model(x) for x in X])\n",
    "\n",
    "    def fit(self, X, y):\n",
    "        \"\"\"\n",
    "        Entrena el modelo buscando el umbral óptimo (brute force) \n",
    "        en un rango de 0 a 30 (número de características).\n",
    "        \"\"\"\n",
    "        best_accuracy = 0\n",
    "        best_threshold = 0\n",
    "        \n",
    "        # El dataset tiene 30 características, el rango de umbrales es [0, 30]\n",
    "        for t in range(X.shape[1] + 1):\n",
    "            self.threshold = t\n",
    "            Y_pred = self.predict(X)\n",
    "            current_accuracy = accuracy_score(y, Y_pred)\n",
    "            \n",
    "            if current_accuracy > best_accuracy:\n",
    "                best_accuracy = current_accuracy\n",
    "                best_threshold = t\n",
    "        \n",
    "        self.threshold = best_threshold\n",
    "        print(f\"Umbral óptimo encontrado: {self.threshold} con accuracy: {best_accuracy:.4f}\")\n",
    "\n",
    "\n",
    "# ==========================================================\n",
    "# 2. Flujo de Entrenamiento y Exportación\n",
    "# ==========================================================\n",
    "print(\"Iniciando proceso de entrenamiento y exportación...\")\n",
    "\n",
    "# Cargar y preparar datos\n",
    "data = load_breast_cancer()\n",
    "X = data.data\n",
    "y = data.target\n",
    "\n",
    "# 2.1 Calcular las medias para la binarización (se usa TODO el dataset)\n",
    "# Esto garantiza que el preprocesamiento de la API sea consistente\n",
    "feature_means = np.mean(X, axis=0)\n",
    "print(f\"Medias de características calculadas. Shape: {feature_means.shape}\")\n",
    "\n",
    "\n",
    "# 2.2 Binarizar X\n",
    "# Convertir los valores continuos a binarios (0 o 1) para la MPNeuron\n",
    "X_bin = (X >= feature_means).astype(int)\n",
    "\n",
    "# 2.3 Dividir datos y entrenar\n",
    "X_train_bin, X_test_bin, y_train, y_test = train_test_split(\n",
    "    X_bin, y, test_size=0.25, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "# 2.4 Instanciar y Entrenar el modelo\n",
    "model = MPNeuron()\n",
    "model.fit(X_train_bin, y_train) \n",
    "# NOTA: Tu notebook encontró un umbral de 27. Este script lo encontrará también.\n",
    "\n",
    "# ==========================================================\n",
    "# 3. Exportar Archivos PKL\n",
    "# ==========================================================\n",
    "\n",
    "# 3.1 Exportar el Modelo Entrenado\n",
    "joblib.dump(model, 'mp_neuron_model.pkl') \n",
    "\n",
    "# 3.2 Exportar las Medias de las Características\n",
    "joblib.dump(feature_means, 'feature_means.pkl')\n",
    "\n",
    "print(\"\\n--- ¡PROCESO TERMINADO! ---\")\n",
    "print(\"Archivos 'mp_neuron_model.pkl' y 'feature_means.pkl' creados en este directorio.\")\n",
    "print(\"Ahora puede ejecutar 'python app.py'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0a918187-2059-4d47-9a3f-8a4c9f851556",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
