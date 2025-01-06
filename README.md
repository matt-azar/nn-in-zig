Written in zig version 0.13. Will probably be deprecated in later versions.

The network currently does not have the option to save or load parameters from a file, so it trains a new network from scratch each time. With its current dimensions, it runs 10 training epochs in ~22 seconds, and performs with ~96.5% accuracy on the test set.
