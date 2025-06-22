Project for the "**Symbolic and Evolutionary Artificial Intelligence**" class (2024-25) at Pisa University.<br>
Group work carried out by the students: [Luca Arduini](https://github.com/LucaArduini) and [Filippo Lucchesi](https://github.com/FilippoLucchesi).

# MLP_FPGA
This project implements a **Multilayer Perceptron (MLP)** on FPGA using modular and reusable Verilog components.  
The design is organized into a **top-level module** and a set of **IP cores**.
## 📁 Project Structure

MLP_FPGA/
├── MLP_IP/
│ ├── MLP_layer_hidden.v # Implementation of the hidden layer
│ ├── MLP_layer_output.v # Implementation of the output layer
│ ├── MLP_mac.v # Multiply-Accumulate (MAC) unit
│ ├── MLP_weight_mem.v # MLP weights memory (ROM)
│ └── ... # Other reusable modules
└── MLP_main.v # Top-level module of the MLP system

## 📌 Component Overview

### 🔹 `MLP_main.v`
- **Top-level module** of the system.
- Instantiates and connects all IP modules.
- Manages data flow and control for inference.

### 🔹 `MLP_IP/` Folder
Contains reusable IP modules:

- **`MLP_layer_hidden.v`**  
  Implements the hidden layer computation of the MLP.

- **`MLP_layer_output.v`**  
  Computes the final output of the model.

- **`MLP_mac.v`**  
  Multiply-Accumulate (MAC) unit used in layer operations.

- **`MLP_weight_mem.v`**  
  ROM memory storing pre-trained model weights.

---

## 🚀 Project Goal

The goal is to implement a **feedforward MLP** optimized for hardware inference on FPGA.  
The IP modules are designed to be modular and reusable in other neural network hardware designs.
