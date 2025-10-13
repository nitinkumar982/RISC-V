<div align="center">

# RISC-V Edge AI with VSDSquadron Pro

### Advanced Machine Learning on Resource-Constrained Embedded Systems

[![RISC-V](https://img.shields.io/badge/RISC--V-FE310--G002-blue?style=for-the-badge&logo=riscv)](https://www.sifive.com/)
[![Python](https://img.shields.io/badge/Python-3.x-green?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)

</div>

---
##  Overview

This repository documents an advanced Edge AI project focused on implementing machine learning algorithms from regression to classification (KNN → SVM) on the *VSDSquadron PRO board, powered by a **SiFive FE310-G002 RISC-V SoC*. The entire workflow, from data preprocessing and model training to 8-bit quantization and bare-metal C inference, is optimized for this low-power, resource-constrained microcontroller.

This work is part of the *VLSI System Design (VSD) RISC-V Edge AI with VSDSquadron Pro* course. It demonstrates the complete machine learning pipeline from regression fundamentals to advanced classification algorithms, all deployed under the strict memory and processing limitations of an embedded system (16KB of SRAM), showcasing a complete "lab-to-fabric" Edge AI implementation.

---

##  Project Objective

The primary goal is to build a complete, embedded system for machine learning classification. The objectives are to:

1. *Master Regression Fundamentals:* Train polynomial regression models to understand non-linear relationships in data.
2. *Progress to Classification:* Implement K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) classifiers for binary classification tasks.
3. *Optimize for the Edge:* Employ *post-training quantization* to convert models to 8-bit integer format, drastically reducing size for microcontroller deployment.
4. *Deploy on RISC-V:* Implement lightweight inference engines in C to run models on the SiFive FE310-G002, achieving robust classification performance on the edge device.

---

##  The Edge AI Challenge: Fitting ML Models in 16KB of RAM

Deploying machine learning models on microcontrollers like the SiFive FE310-G002 is non-trivial. A standard floating-point model can be several megabytes in size, while the target board has only *16KB of SRAM*. This project overcomes this challenge through:

* *Aggressive Quantization:* Converting all model parameters from 32-bit floating-point to 8-bit integers. This reduces the model size by ~4x.
* *Optimized C-Inference:* Writing a minimal, bare-metal C function for inference that performs all calculations using integer arithmetic, avoiding the need for heavy floating-point libraries.
* *Feature Engineering:* Preprocessing input data to reduce dimensionality while maintaining classification accuracy.

---

##  Methodology & Deployment Pipeline

The project follows a structured pipeline from Python-based model development to C-based firmware deployment.

### 1️⃣ Data Preprocessing & Model Training (Python)

To build robust models, the dataset was carefully prepared using Python:

* *Polynomial Regression:* Implemented non-linear regression to model relationships between features and targets, visualizing fitted curves against actual data points.
* *Classification Dataset:* Analyzed age vs. estimated salary data with binary classification (classes 0 and 1).
* *Data Normalization:* Scaled features for better model convergence and generalization.

<p align="center">
  <img src="images/download.png" alt="Linear Regression" width="500"/>
  <br>
  <em>Linear Regression Analysis</em>
</p>

### 2️⃣ Classification Models (Scikit-learn)

* *K-Nearest Neighbors (KNN):* Implemented distance-based classification with visualization of decision boundaries.
* *Support Vector Machine (SVM):* Explored multiple kernel functions:
  - *Linear Kernel:* For linearly separable data
  - *Polynomial Kernel:* For capturing curved decision boundaries
  - *RBF Kernel:* For highly non-linear classification tasks

<p align="center">
  <img src="images/download(4).png" alt="KNN Classification" width="500"/>
  <br>
  <em>K-Nearest Neighbors Classification</em>
</p>

<p align="center">
  <img src="images/download(7).png" alt="Confusion Matrix" width="500"/>
  <br>
  <em>Model Performance - Confusion Matrix</em>
</p>

<p align="center">
  <img src="images/download(5).png" alt="RBF SVM Training" width="500"/>
  <br>
  <em>RBF Kernel SVM Training Results</em>
</p>

### 3️⃣ Neural Networks on RISC-V Microcontrollers

* *MLP Classifier:* Built Multi-Layer Perceptron with hidden layers (64, 32), ReLU activation, and Adam optimizer.
* *Training Performance:* Achieved *96% accuracy* on 60,000 training samples with 784 features.
* *Model Parameters:* 52,650 total trainable parameters.

### 4️⃣ Post-Training Quantization (TensorFlow Lite)

* *Model Conversion:* After training, models were converted to TensorFlow Lite format and quantized to 8-bit integer (INT8) representation.
* *Size Reduction:* Critical step for reducing model size from megabytes to kilobytes suitable for embedded deployment.

### 5️⃣ C-Code Generation & Bare-Metal Inference (RISC-V)

* *Model to C Array:* Converted .tflite model into C header and source files (model_params.h, model_params.c) containing quantized weights and biases as integer arrays.
* *Lightweight Inference Engine:* Wrote simple C functions to perform neural network operations using only integer arithmetic on the RISC-V core.

---

## 💻 Hardware and Software

<table>
<tr>
<td width="50%">

#### 🔧 Target Hardware

* *Board:* VSDSquadron PRO
* *SoC:* SiFive FE310-G002 (32-bit RV32IMAC instruction set)
* *Memory:* 16KB SRAM, 128Mbit QSPI Flash

</td>
<td width="50%">

#### 🖥 Development Environment

* *AI/ML:* Python 3.x, scikit-learn, TensorFlow, NumPy, Matplotlib
* *Embedded:* RISC-V GNU Toolchain, Freedom Studio
* *Tools:* xxd, zadig-2.9.exe

</td>
</tr>
</table>

---

##  Getting Started

### Part A: Training the Model (Python Environment)

1. *Clone the repository:*
bash
   git clone https://github.com/nitinkumar982/RISC-V.git
   cd RISC-V

2. *Setup Python Environment:*
    bash
   cd python_training
   pip install -r requirements.txt

3. **Download Dataset:**

Prepare your classification dataset (age vs. salary or similar binary classification data).
Place it in the data/ directory.

#### Run Training & Quantization:

Open and run the Jupyter notebooks in sequence to train models, quantize them, and generate C-arrays.


### Part B: Deploying to RISC-V Board

#### 1. Open Freedom Studio and Import the Project

Launch Freedom Studio.
Go to File > Import... > Existing Projects into Workspace.
Select the project directory and import it.

#### 2. Clean and Build the Project

Navigate to:

Project > Clean... — select the project and confirm.
Project > Build Project — this will compile the project and generate the main.elf file.

#### 3. Run in Simulation (Optional)

Configure a debug launch for simulation:

Go to Run > Debug Configurations....
Create or modify a SiFive GDB OpenOCD Debugging configuration.
Ensure the Executable path points to the correct main.elf file.

Start the simulation.
Observe output in the Freedom Studio Console.

#### 4. Flash to VSDSquadron PRO Board

Connect the board via USB.
If on Windows, use zadig-2.9.exe to install the correct USB drivers:

Run Zadig.
Select the connected USB device (usually shows as "USB Serial" or similar).
Choose WinUSB as the driver and click Install Driver.


Use your preferred flashing tool, such as OpenOCD, to flash the main.elf file:
```
openocd -f interface/ftdi/vsdsquadron.cfg -f target/sifive.cfg -c "program path/to/main.elf verify reset exit"
```

## Key Learnings
This project provides practical experience in the complete end-to-end workflow of an Edge AI application:

#### * Model Progression: Mastered the journey from polynomial regression to advanced SVM and neural network classifiers.
#### * Model Optimization: Applied post-training quantization to fit 52,650-parameter models onto tiny devices.
#### * Bare-Metal Programming: Understood how to implement ML inference from scratch in C for a RISC-V target without an OS.


## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

* Thanks to VLSI System Design (VSD) for providing the course resources and project guidance.
* Thanks to SiFive For the FE310-G002 SoC specifications.
* Credit to scikit-learn and TensorFlow teams for their powerful machine learning frameworks.
