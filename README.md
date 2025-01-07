# Neural Network in Zig #

Written in zig version 0.13. Will probably be deprecated in later versions.

The network is trained on the standard MNIST training set with 60,000 handwritten digits in 28 x 28 pixel images, and it tests on the standard MNIST test set with 10,000 more handwritten digits. With its current dimensions, on my laptop, it runs 10 training epochs in ~20 seconds and performs with ~96.5% accuracy on the test set.
