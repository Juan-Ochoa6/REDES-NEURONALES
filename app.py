from flask import Flask, render_template, request
import tensorflow as tk
import numpy as np

app = Flask(__name__)

# =====================================
# CARGAR MODELO
# =====================================

modelo = tk.keras.models.load_model("hipotenusa.h5")

# =====================================
# PAGINA PRINCIPAL
# =====================================

@app.route("/", methods=["GET", "POST"])
def inicio():

    resultado = ""

    if request.method == "POST":

        a = float(request.form["catetoA"])
        b = float(request.form["catetoB"])

        prediccion = modelo.predict(
            np.array([[a/100.0, b/100.0]]),
            verbose=False
        )

        resultadoReal = prediccion[0][0] * 150.0

        resultado = round(resultadoReal, 2)

    return render_template(
        "index.html",
        resultado=resultado
    )

# =====================================
if __name__ == "__main__":
    app.run(debug=True)