import tensorflow as tk
import numpy as np

# =========================================
# GENERAR DATOS
# =========================================

catetoA = []
catetoB = []
hipotenusa = []

for a in range(1, 51):
    for b in range(1, 51):

        h = np.sqrt(a**2 + b**2)

        catetoA.append(a)
        catetoB.append(b)
        hipotenusa.append(h)

# Convertir a arrays
catetoA = np.array(catetoA, dtype=float)
catetoB = np.array(catetoB, dtype=float)
hipotenusa = np.array(hipotenusa, dtype=float)

# =========================================
# NORMALIZAR
# =========================================

catetoA = catetoA / 100.0
catetoB = catetoB / 100.0
hipotenusa = hipotenusa / 150.0

# =========================================
# RED NEURONAL
# =========================================

capa1 = tk.keras.layers.Dense(
    units=16,
    input_shape=[2],
    activation='relu'
)

capa2 = tk.keras.layers.Dense(
    units=16,
    activation='relu'
)

capa3 = tk.keras.layers.Dense(units=1)

modelo = tk.keras.Sequential([
    capa1,
    capa2,
    capa3
])

# =========================================
# COMPILAR
# =========================================

modelo.compile(
    optimizer=tk.keras.optimizers.Adam(0.001),
    loss='mean_squared_error'
)

# =========================================
# ENTRENAMIENTO
# =========================================

print("Entrenando IA...")

modelo.fit(
    np.column_stack((catetoA, catetoB)),
    hipotenusa,
    epochs=200,
    verbose=True
)

print("IA entrenada!")

# =========================================
# GUARDAR MODELO
# =========================================

modelo.save("hipotenusa.h5")

print("Modelo guardado!")