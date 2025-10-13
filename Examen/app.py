{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5a82ee3b-db93-4feb-a465-f6a6306a7db1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Éxito: Modelo y medias cargadas. Servidor listo.\n",
      "Umbral usado por el modelo: 0\n",
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      " * Restarting with stat\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[31mSystemExit\u001b[39m\u001b[31m:\u001b[39m 1\n"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "import numpy as np\n",
    "from flask import Flask, request, jsonify\n",
    "from flask_cors import CORS\n",
    "\n",
    "# ==========================================================\n",
    "# 1. Definición de la Clase del Modelo\n",
    "# NOTA: Esta definición es OBLIGATORIA para que joblib pueda\n",
    "# reconstruir el objeto MPNeuron serializado desde el archivo .pkl\n",
    "# ==========================================================\n",
    "class MPNeuron:\n",
    "    \"\"\"Clase que implementa la Neurona de McCulloch y Pitts.\"\"\"\n",
    "    def __init__(self, threshold=0):\n",
    "        self.threshold = threshold\n",
    "        \n",
    "    def model(self, x):\n",
    "        \"\"\"Función de activación: 1 si la suma >= umbral, 0 en caso contrario.\"\"\"\n",
    "        # La neurona MP suma las entradas (ya binarizadas)\n",
    "        return (np.sum(x) >= self.threshold).astype(int)\n",
    "    \n",
    "    def predict(self, X):\n",
    "        \"\"\"Genera predicciones para un conjunto de datos X.\"\"\"\n",
    "        return np.array([self.model(x) for x in X])\n",
    "\n",
    "\n",
    "# ==========================================================\n",
    "# 2. CONFIGURACIÓN INICIAL DEL SERVIDOR FLASK\n",
    "# ==========================================================\n",
    "app = Flask(__name__)\n",
    "# Habilitar CORS para permitir peticiones desde cualquier origen (necesario para el front-end local)\n",
    "CORS(app) \n",
    "\n",
    "# --- 3. CARGAR MODELO Y MEDIAS ---\n",
    "mp_model = None\n",
    "feature_means = None\n",
    "\n",
    "try:\n",
    "    # Cargar la clase MPNeuron con su umbral óptimo (t=0 o t=27, dependiendo del entrenamiento)\n",
    "    mp_model = joblib.load('mp_neuron_model.pkl')\n",
    "    # Cargar las medias (vector de 30 valores) usadas para binarizar los nuevos datos\n",
    "    feature_means = joblib.load('feature_means.pkl')\n",
    "    print(\"Éxito: Modelo y medias cargadas. Servidor listo.\")\n",
    "    print(f\"Umbral usado por el modelo: {mp_model.threshold}\")\n",
    "except Exception as e:\n",
    "    print(f\"ERROR: No se pudieron cargar los archivos del modelo: {e}\")\n",
    "    mp_model = None\n",
    "    feature_means = None\n",
    "\n",
    "\n",
    "# ==========================================================\n",
    "# 4. ENDPOINT DE PREDICCIÓN\n",
    "# ==========================================================\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    # Verificar que el modelo se cargó correctamente\n",
    "    if not mp_model or feature_means is None:\n",
    "        return jsonify({'error': 'Modelo no disponible en el servidor'}), 500\n",
    "\n",
    "    try:\n",
    "        # 1. Recibir los 30 valores continuos del frontend (JSON)\n",
    "        data = request.get_json()\n",
    "        \n",
    "        # El frontend debe enviar un JSON con 30 claves (ej. f1, f2, ..., f30)\n",
    "        # Convertir los valores a un array de NumPy (30 valores continuos)\n",
    "        input_values = np.array(list(data.values())).astype(float)\n",
    "        \n",
    "        if input_values.shape[0] != 30:\n",
    "            return jsonify({'error': f'Se esperaban 30 características, se recibieron {input_values.shape[0]}.'}), 400\n",
    "\n",
    "        # 2. Paso CRÍTICO: Binarizar la entrada usando las medias guardadas\n",
    "        # La lógica es: Valor >= Media -> 1, sino -> 0\n",
    "        input_data_bin = (input_values >= feature_means).astype(int)\n",
    "        \n",
    "        # 3. Realizar la predicción\n",
    "        # El modelo espera una matriz 2D (una fila, 30 columnas)\n",
    "        input_data_bin = input_data_bin.reshape(1, -1) \n",
    "        prediction = mp_model.predict(input_data_bin)[0]\n",
    "\n",
    "        # 4. Devolver la respuesta al frontend\n",
    "        resultado_texto = 'Benigno' if prediction == 1 else 'Maligno'\n",
    "        \n",
    "        return jsonify({\n",
    "            'prediction': int(prediction),\n",
    "            'diagnosis': resultado_texto,\n",
    "            'status': 'success'\n",
    "        })\n",
    "\n",
    "    except Exception as e:\n",
    "        # Manejo de errores de formato, conversión o procesamiento\n",
    "        return jsonify({'error': f'Error en el procesamiento de datos: {str(e)}'}), 400\n",
    "\n",
    "# ==========================================================\n",
    "# 5. INICIAR EL SERVIDOR\n",
    "# ==========================================================\n",
    "if __name__ == '__main__':\n",
    "    # Ejecuta el servidor en http://127.0.0.1:5000/\n",
    "    app.run(debug=True, port=5000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "86ff689f-7f96-4911-a375-beecf6038e27",
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
