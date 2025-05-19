import math

# --- Configuration for Q8.8 ---
TOTAL_BITS = 16
FRACTIONAL_BITS = 8
INTEGER_BITS = TOTAL_BITS - FRACTIONAL_BITS # Should be 8

SCALE_FACTOR = 2**FRACTIONAL_BITS  # 2^8 = 256

# Calculate min and max integer values for a signed 16-bit number
MIN_INT_VAL = -(2**(TOTAL_BITS - 1))  # -32768
MAX_INT_VAL = (2**(TOTAL_BITS - 1)) - 1 #  32767

def float_to_q8_8_hex(float_val):
    """Converts a float to its Q8.8 16-bit 2's complement hex representation."""
    # Scale the float
    scaled_val = float_val * SCALE_FACTOR

    # Round to nearest integer
    rounded_int_val = int(round(scaled_val))

    # Clamp/Saturate the integer value
    clamped_int_val = max(MIN_INT_VAL, min(MAX_INT_VAL, rounded_int_val))

    # Convert to 16-bit 2's complement hex
    # If the number is negative, Python's hex() gives "-0x...", which is not what we want.
    # We need the bit pattern of the 2's complement number.
    # The `& ((1 << TOTAL_BITS) - 1)` operation ensures we get the correct
    # bit pattern for negative numbers when interpreted as unsigned for hex formatting.
    # Example: -1 (int) & 0xFFFF -> 0xFFFF (unsigned representation of -1 in 16-bit 2's comp)
    hex_representation = format(clamped_int_val & ((1 << TOTAL_BITS) - 1), f'0{TOTAL_BITS//4}X')
    return hex_representation

def float_to_q8_8_bin(float_val):
    """Converts a float to its Q8.8 16-bit 2's complement binary representation."""
    scaled_val = float_val * SCALE_FACTOR
    rounded_int_val = int(round(scaled_val))
    clamped_int_val = max(MIN_INT_VAL, min(MAX_INT_VAL, rounded_int_val))
    bin_representation = format(clamped_int_val & ((1 << TOTAL_BITS) - 1), f'0{TOTAL_BITS}b')
    return bin_representation

def process_weight_file(input_filepath, output_filepath, output_format="hex"):
    """
    Reads floats from input_filepath, converts them to Q8.8,
    and writes them to output_filepath, one per line.
    """
    print(f"Processing {input_filepath} -> {output_filepath} (format: {output_format})")
    count = 0
    with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
        if output_format == "hex":
            outfile.write(f"// Q8.8 Hex Weights (Total Bits: {TOTAL_BITS}, Fractional Bits: {FRACTIONAL_BITS})\n")
            outfile.write(f"// Scaled by 2^{FRACTIONAL_BITS} = {SCALE_FACTOR}\n")
            outfile.write(f"// Integer Range: [{MIN_INT_VAL}, {MAX_INT_VAL}]\n")
            outfile.write(f"// Float Range: [{MIN_INT_VAL/SCALE_FACTOR:.4f}, {MAX_INT_VAL/SCALE_FACTOR:.4f}]\n")
        elif output_format == "bin":
            outfile.write(f"// Q8.8 Binary Weights (Total Bits: {TOTAL_BITS}, Fractional Bits: {FRACTIONAL_BITS})\n")
            # ... (add similar header comments for binary if desired)

        for line_num, line in enumerate(infile):
            parts = line.strip().split()
            for i, part_str in enumerate(parts):
                if not part_str:  # Skip empty strings if there are multiple spaces
                    continue
                try:
                    float_val = float(part_str)
                    if output_format == "hex":
                        converted_val = float_to_q8_8_hex(float_val)
                    elif output_format == "bin":
                        converted_val = float_to_q8_8_bin(float_val)
                    else:
                        raise ValueError("Invalid output_format specified.")

                    original_val_comment = f"{float_val:.8f}" # Original float value for reference
                    outfile.write(f"{converted_val} // Original: {original_val_comment}, Input Line: {line_num+1}, Item: {i+1}\n")
                    count +=1
                except ValueError:
                    print(f"Warning: Could not convert '{part_str}' on line {line_num+1} of {input_filepath}. Skipping.")
    print(f"Successfully converted {count} weights.\n")

# --- Main execution ---
if __name__ == "__main__":
    # --- Process weights_w1.txt ---
    # Output in HEX format
    process_weight_file('weights_w1.txt', "weights_w1.hex", output_format="hex")
    # Optionally, output in BINARY format
    # process_weight_file("weights_w1.txt", "weights_w1.bin", output_format="bin")

    # --- Process weights_w2.txt ---
    # Output in HEX format
    process_weight_file('weights_w2.txt', "weights_w2.hex", output_format="hex")
    # Optionally, output in BINARY format
    # process_weight_file("weights_w2.txt", "weights_w2.bin", output_format="bin")

    print("Conversion complete.")

    # --- Example of how to interpret back (for verification) ---
    print("\n--- Verification Examples (Q8.8) ---")
    test_floats = [-0.30881676, 2.71299577, 0.01931785, -128.0, 127.99, 0.0, 0.001, -0.001]
    for tf in test_floats:
        hex_val = float_to_q8_8_hex(tf)
        bin_val = float_to_q8_8_bin(tf)
        # To convert back from the 16-bit integer representation:
        # First, parse hex to integer. If it's > MAX_INT_VAL, it's negative in 2's complement.
        int_from_hex = int(hex_val, 16)
        if int_from_hex > MAX_INT_VAL: # Handle 2's complement negative
            signed_int_val = int_from_hex - (1 << TOTAL_BITS)
        else:
            signed_int_val = int_from_hex
        reconstructed_float = signed_int_val / SCALE_FACTOR
        print(f"Float: {tf:>12.8f} -> Scaled & Rounded Int: {int(round(tf*SCALE_FACTOR)):>6} -> Hex: {hex_val} -> Bin: {bin_val} -> Reconstructed: {reconstructed_float:>12.8f}")