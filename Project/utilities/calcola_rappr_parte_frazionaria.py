def frazione_binaria_16bit(numero):
    if not (numero < 1):
        raise ValueError("Il numero deve essere compreso tra 0 e 1 (esclusi)")

    risultato = ""
    frazione = numero

    for _ in range(16):
        frazione *= 2
        bit = int(frazione)
        risultato += str(bit)
        frazione -= bit

    # Gruppi di 4 bit
    gruppi = [risultato[i:i+4] for i in range(0, 16, 4)]
    risultato_formattato = ' '.join(gruppi)

    # Conversione in esadecimale
    hex_gruppi = [hex(int(gruppo, 2))[2:].upper() for gruppo in gruppi]  # [2:] per togliere '0x'

    print(f"Rappresentazione binaria (16 bit): {risultato_formattato}")
    print(f"Rappresentazione esadecimale dei gruppi: {' '.join(hex_gruppi)}")

    return risultato_formattato

# Esempio d'uso
numero = 0.0044
frazione_binaria_16bit(numero)
