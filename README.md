# Neural Network in Zig #

Written in zig version 0.13. Will probably be deprecated in later versions.

The network is trained on the standard MNIST training set with 60,000 handwritten digits in 28 x 28 pixel images, and it tests on the standard MNIST test set with 10,000 more handwritten digits. With its current dimensions, on my laptop, it runs 10 training epochs in ~20 seconds and performs with ~96.5% accuracy on the test set.

To build and run, open a terminal in the root directory of the project and run the following command:

```bash
zig build run
```

You'll be prompted to load the model from a file. Hit enter to continue and the program will train and then save the model to a file mnist_model.bin, which you can load in the future.

After the model is trained or loaded, you'll be prompted to draw a digit in a GUI window. Hit enter to submit the drawing, and the program will classify it and print the result in the terminal.
