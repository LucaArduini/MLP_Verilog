import numpy as np
import random

def sinc2D(x, y):
  """
  Calcola la funzione sinc non normalizzata bidimensionale: sinc(x) * sinc(y)
  dove sinc(t) = sin(t) / t per t != 0, e 1 per t = 0.

  Args:
    x: Il valore x per la funzione sinc.
    y: Il valore y per la funzione sinc.

  Returns:
    Il valore di sinc(x) * sinc(y).
  """

  # Calcola sinc(x)
  if x == 0:
    sinc_x = 1.0
  else:
    sinc_x = np.sin(x) / x  # Modifica: rimossa la moltiplicazione per pi

  # Calcola sinc(y)
  if y == 0:
    sinc_y = 1.0
  else:
    sinc_y = np.sin(y) / y  # Modifica: rimossa la moltiplicazione per pi

  return sinc_x * sinc_y

# --- Generazione di dati di test casuali ---
num_test_pairs = 100
test_data_pairs = []

for _ in range(num_test_pairs):
  random_x = random.uniform(-10, 10)
  random_y = random.uniform(-10, 10)
  test_data_pairs.append((random_x, random_y))

# --- Esecuzione dei test e stampa dei risultati ---
print(f"--- Calcolo di sinc2D(x,y) = (sin(x)/x) * (sin(y)/y) per {num_test_pairs} coppie casuali ---")

# Stampa anche alcuni casi specifici per verifica
print("\nCasi specifici:")
# Caso 1: Entrambi x e y sono diversi da zero
x_val1 = np.pi / 2  # sin(pi/2) = 1, x = pi/2 -> sinc_x = 1 / (pi/2) = 2/pi
y_val1 = np.pi      # sin(pi) = 0, y = pi   -> sinc_y = 0 / pi = 0
risultato1 = sinc2D(x_val1, y_val1)
print(f"sinc2D({x_val1:.4f}, {y_val1:.4f}) = {risultato1:.6f} (Atteso: 2/pi * 0 = 0)")

# Caso 2: x è zero, y è diverso da zero
x_val2 = 0.0
y_val2 = 5.0
risultato2 = sinc2D(x_val2, y_val2)
print(f"sinc2D({x_val2:.4f}, {y_val2:.4f}) = {risultato2:.6f} (Atteso: 1 * sin(5)/5)")
print(f"Valore atteso per y: {np.sin(5.0)/5.0:.6f}")


# Caso 3: Entrambi x e y sono zero
x_val4 = 0.0
y_val4 = 0.0
risultato4 = sinc2D(x_val4, y_val4)
print(f"sinc2D({x_val4:.4f}, {y_val4:.4f}) = {risultato4:.6f} (Atteso: 1 * 1 = 1)")


print("\nTest con coppie generate casualmente:")
for i, (x_val, y_val) in enumerate(test_data_pairs):
  risultato = sinc2D(x_val, y_val)
  # Limita la stampa per non avere un output eccessivo se num_test_pairs è molto grande
  if i < 20 or i >= num_test_pairs - 5 : # Stampa i primi 20 e gli ultimi 5
    print(f"Test {i+1:3d}: sinc2D({x_val:9.4f}, {y_val:9.4f}) = {risultato:.6f}")
  elif i == 20:
    print("...")

print(f"\n--- Test completati. Elaborate {len(test_data_pairs)} coppie. ---")

# print sinc2D(-3,7)
print(f"\nsinc2D(-3,7) = {sinc2D(-3,7)}")