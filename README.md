# mlperceptron
Work-in-progress vanilla implementation of a `fully connected layers neural network` (`multilayer perceptron`) written in C and developed as an external object for the `Pure Data`.

The project aims to integrate machine learning algorithms into the `Pure Data` environment and to provide a configurable neural network architecture (with no external machine learning dependencies) designed for technical studies and real-time composition applications.



# Key Features
✅ Fully configurable network architecture

Define the number of layers, input dimension, and neurons per layer.


🧩 Dynamically allocated matrices and vectors

``Weight matrices``

``Bias vectors``

``Activation vectors``

``Z vectors``

``Delta vectors``



📥 Training parameters

Accepts ``input data`` and ``labels`` as lists

Accepts float for ``learning rate``

Accepts an integer for the ``number of epochs``

``Training mode`` toggle (0 = OFF, 1 = ON)

💾 Model I/O

Saves trained model to a ``.txt file``

Loads trained model from ``.txt file``

⚙️ ``Weight`` & ``bias`` initialization methods

``Random``

``Zeros``

``Uniform``

``He``

``Xavier``

``Lecun``

``Custom range``

🧠 Activation functions supported per layer

``sigmoid``

``relu``

``tanh``

``prelu``

``softmax``

``softplus``


# Build
> [!NOTE]
`mlperceptron` uses `pd.build`. To build the external on Linux, Mac, and Windows (using Mingw64):

1. `git clone https://github.com/oviniciuscesar/mlperceptron/ --recursive`;
2. `cd cnn2d`;
4. `cmake . -B build`;
5. `cmake --build build`;
