# Neural Network in Zig #

Written in zig version 0.14.

The program implements a simple neural network to classify handwritten digits from the MNIST dataset. It includes a primitive GUI for configuring the neural network's training options and it gives the user the option to draw digits in a popup window for classification. Results are printed to the console.

The program was written and tested on Linux. I don't know if it will work without modifications on Windows or macOS.

To build and run, open a terminal in the root directory of the project and run the following command:

```bash
zig build run
```

Upon running the program, a GUI window will appear where you can select a few training parameters for the neural network. You also have the option to load the weights and biases from a pre-trained model. Currently the only option is to load from the default location "mnist_model.bin", which is also the default save location for the model after training. Every time you train the model, it will overwrite this file. Training the model takes less than a minute with a reasonable number of epochs (each epoch takes about 2 seconds on my machine).
