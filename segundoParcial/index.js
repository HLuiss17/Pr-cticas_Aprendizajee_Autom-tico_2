const API_URL = "http://127.0.0.1:5000/predict";

document.getElementById("predictBtn").addEventListener("click", async () => {
    const inputStr = document.getElementById("inputValues").value;
    
    // Convertimos la cadena "1,0,0,0" en un arreglo de números [1,0,0,0]
    const inputArray = inputStr.split(",").map(Number);

    // Puedes enviar múltiples vectores, aquí usamos solo uno
    const dataToSend = [inputArray];

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ inputs: dataToSend })
        });
        const data = await response.json();

        document.getElementById("result").innerHTML = 
            "Predicción: " + data.predictions.join(", ");
    } catch (error) {
        document.getElementById("result").innerHTML = "Error: " + error;
        console.error(error);
    }
});
