const std = @import("std");
const mnist_loader = @import("mnist_loader.zig");
const stdout = std.io.getStdOut().writer();

/// Neural network with two hidden layers.
/// Equipped with methods for initialization, stochastic gradient descent,
/// backpropagation, softmax, cost,
/// and saving/loading parameters to/from a file.
pub const Network = struct {
    pub const image_size = 784;
    pub const hidden_size1 = 32;
    pub const hidden_size2 = 64;
    pub const output_size = 10;

    weights1: [image_size * hidden_size1]f64,
    weights2: [hidden_size1 * hidden_size2]f64,
    weights3: [hidden_size2 * output_size]f64,
    biases1: [hidden_size1]f64,
    biases2: [hidden_size2]f64,
    biases3: [output_size]f64,
    hidden1: [hidden_size1]f64,
    hidden2: [hidden_size2]f64,
    output: [output_size]f64,

    /// Returns a network with all weights and biases set to zero.
    pub fn init() Network {
        const net = Network{
            .weights1 = [_]f64{0} ** (image_size * hidden_size1),
            .weights2 = [_]f64{0} ** (hidden_size1 * hidden_size2),
            .weights3 = [_]f64{0} ** (hidden_size2 * output_size),
            .biases1 = [_]f64{0} ** hidden_size1,
            .biases2 = [_]f64{0} ** hidden_size2,
            .biases3 = [_]f64{0} ** output_size,
            .hidden1 = [_]f64{0} ** hidden_size1,
            .hidden2 = [_]f64{0} ** hidden_size2,
            .output = [_]f64{0} ** output_size,
        };

        return net;
    }

    /// Initialize weights using Xavier initialization.
    pub fn initializeWeights(self: *Network) void {
        const limit1 = std.math.sqrt(6.0 / @as(f64, Network.image_size));
        const limit2 = std.math.sqrt(6.0 / @as(f64, Network.hidden_size1));
        const limit3 = std.math.sqrt(6.0 / @as(f64, Network.hidden_size2));

        var rng = std.rand.DefaultPrng.init(@abs(std.time.timestamp()));

        for (self.weights1[0..]) |*w| {
            w.* = rng.random().float(f64) * (2.0 * limit1) - limit1;
        }

        for (self.weights2[0..]) |*w| {
            w.* = rng.random().float(f64) * (2.0 * limit2) - limit2;
        }

        for (self.weights3[0..]) |*w| {
            w.* = rng.random().float(f64) * (2.0 * limit3) - limit3;
        }
    }

    /// Initialize biases to a random value in [0, 0.01].
    pub fn initializeBiases(self: *Network) void {
        var rng = std.rand.DefaultPrng.init(@abs(std.time.timestamp()));

        for (self.biases1[0..]) |*b| {
            b.* = rng.random().float(f64) / 100;
        }

        for (self.biases2[0..]) |*b| {
            b.* = rng.random().float(f64) / 100;
        }

        for (self.biases3[0..]) |*b| {
            b.* = rng.random().float(f64) / 100;
        }
    }

    /// Softmax activation function.
    pub fn softmax(self: *Network) void {
        var max = self.output[0];
        for (self.output[1..]) |x| {
            if (x > max) max = x;
        }

        var sum: f64 = 0.0;

        for (self.output[0..]) |*x| {
            x.* = std.math.exp(x.* - max);
            sum += x.*;
        }

        for (self.output[0..]) |*x| {
            x.* /= sum;
        }
    }

    /// Compute the loss using the cross-entropy cost function.
    pub fn cost(self: *Network, target_label: u8) f64 {
        const epsilon = 1e-9; // Small value to prevent log(0)
        return -std.math.log(f64, std.math.e, self.output[target_label] + epsilon);
    }

    /// Train the neural network using gradient descent.
    pub fn forward(self: *Network, input: []f64, dropout_rate: f64, is_training: bool) void {
        var rng = std.rand.DefaultPrng.init(@abs(std.time.timestamp()));

        // First hidden layer
        for (0..Network.hidden_size1) |i| {
            var sum: f64 = self.biases1[i];
            for (0..Network.image_size) |j| {
                sum += input[j] * self.weights1[i * Network.image_size + j];
            }
            self.hidden1[i] = relu(sum);
            if (is_training) {
                self.hidden1[i] *= if (rng.random().float(f64) > dropout_rate) 1.0 else 0.0;
            }
        }

        // Second hidden layer
        for (0..Network.hidden_size2) |i| {
            var sum: f64 = self.biases2[i];
            for (0..Network.hidden_size1) |j| {
                sum += self.hidden1[j] * self.weights2[i * Network.hidden_size1 + j];
            }
            self.hidden2[i] = relu(sum);
            if (is_training) {
                self.hidden2[i] *= if (rng.random().float(f64) > dropout_rate) 1.0 else 0.0;
            }
        }

        // Output layer
        for (0..Network.output_size) |i| {
            var sum: f64 = self.biases3[i];
            for (0..Network.hidden_size2) |j| {
                sum += self.hidden2[j] * self.weights3[i * Network.hidden_size2 + j];
            }
            self.output[i] = sum;
        }

        self.softmax();
    }

    /// Compute the gradient of the cost function and use it to update the weights and biases.
    pub fn backpropagate(
        self: *Network,
        input: []f64,
        target_label: u8,
        learning_rate: f64,
    ) void {
        var target = [_]f64{0} ** Network.output_size;
        target[target_label] = 1.0;

        var output_error = [_]f64{0} ** Network.output_size;
        var output_delta = [_]f64{0} ** Network.output_size;
        var hidden2_error = [_]f64{0} ** Network.hidden_size2;
        var hidden2_delta = [_]f64{0} ** Network.hidden_size2;
        var hidden1_error = [_]f64{0} ** Network.hidden_size1;
        var hidden1_delta = [_]f64{0} ** Network.hidden_size1;

        // Output layer errors and deltas
        for (0..Network.output_size) |i| {
            output_error[i] = target[i] - self.output[i];
            output_delta[i] = output_error[i];
            self.biases3[i] += learning_rate * output_delta[i];
        }

        // Hidden layer 2 errors and deltas
        for (0..Network.hidden_size2) |i| {
            var sum: f64 = 0.0;
            for (0..Network.output_size) |j| {
                sum += output_delta[j] * self.weights3[j * Network.hidden_size2 + i];
            }
            hidden2_error[i] = sum;
            hidden2_delta[i] = hidden2_error[i] * reluDerivative(self.hidden2[i]);
            self.biases2[i] += learning_rate * hidden2_delta[i];
        }

        // Hidden layer 1 errors and deltas
        for (0..Network.hidden_size1) |i| {
            var sum: f64 = 0.0;
            for (0..Network.hidden_size2) |j| {
                sum += hidden2_delta[j] * self.weights2[j * Network.hidden_size1 + i];
            }
            hidden1_error[i] = sum;
            hidden1_delta[i] = hidden1_error[i] * reluDerivative(self.hidden1[i]);
            self.biases1[i] += learning_rate * hidden1_delta[i];
        }

        // Update weights
        for (0..Network.output_size) |i| {
            for (0..Network.hidden_size2) |j| {
                self.weights3[i * Network.hidden_size2 + j] += learning_rate * output_delta[i] * self.hidden2[j];
            }
        }

        for (0..Network.hidden_size2) |i| {
            for (0..Network.hidden_size1) |j| {
                self.weights2[i * Network.hidden_size1 + j] += learning_rate * hidden2_delta[i] * self.hidden1[j];
            }
        }

        for (0..Network.hidden_size1) |i| {
            for (0..Network.image_size) |j| {
                self.weights1[i * Network.image_size + j] += learning_rate * hidden1_delta[i] * input[j];
            }
        }
    }

    /// Return the index of the greatest value in the output layer.
    pub fn predict(self: *Network) usize {
        var max_index: usize = 0;
        var max_value: f64 = self.output[0];

        for (1..Network.output_size) |i| {
            if (self.output[i] > max_value) {
                max_value = self.output[i];
                max_index = i;
            }
        }
        return max_index;
    }

    /// Save the network's parameters to a file in your current working directory.
    pub fn save(self: *Network, file_name: []const u8) !void {
        const file = try std.fs.cwd().createFile(file_name, .{ .truncate = true });
        defer file.close();
        const writer = file.writer();
        var buffer: [64]u8 = undefined;

        const Helper = struct {
            fn writeArray(_writer: anytype, _buffer: []u8, array: []const f64) !void {
                for (array) |value| {
                    const str_value = try std.fmt.formatFloat(_buffer[0..], value, .{ .mode = .decimal });
                    try _writer.writeAll(str_value);
                    try _writer.writeAll("\n");
                }
            }
        };

        try Helper.writeArray(writer, &buffer, self.weights1[0..]);
        try Helper.writeArray(writer, &buffer, self.weights2[0..]);
        try Helper.writeArray(writer, &buffer, self.weights3[0..]);
        try Helper.writeArray(writer, &buffer, self.biases1[0..]);
        try Helper.writeArray(writer, &buffer, self.biases2[0..]);
        try Helper.writeArray(writer, &buffer, self.biases3[0..]);

        try stdout.print("Successfully saved network to {s}\n", .{file_name});
    }

    /// Load the network's parameters from a file in your current working directory.
    pub fn load(self: *Network, file_name: []const u8) !void {
        const file = try std.fs.cwd().openFile(file_name, .{});
        defer file.close();
        const reader = file.reader();
        var buffer: [64]u8 = undefined;

        const Helper = struct {
            fn readArray(_reader: anytype, _buffer: []u8, array: []f64) !void {
                for (array) |*value| {
                    const line = try _reader.readUntilDelimiterOrEof(_buffer[0..], '\n') orelse {
                        return error.UnexpectedEndOfFile;
                    };
                    value.* = try std.fmt.parseFloat(f64, line);
                }
            }
        };

        try Helper.readArray(reader, &buffer, self.weights1[0..]);
        try Helper.readArray(reader, &buffer, self.weights2[0..]);
        try Helper.readArray(reader, &buffer, self.weights3[0..]);
        try Helper.readArray(reader, &buffer, self.biases1[0..]);
        try Helper.readArray(reader, &buffer, self.biases2[0..]);
        try Helper.readArray(reader, &buffer, self.biases3[0..]);

        try stdout.print("Successfully loaded network from {s}\n", .{file_name});
    }
};

/// ReLU activation function.
/// ReLU(x) = max(0, x).
pub fn relu(x: f64) f64 {
    return if (x > 0.0) x else 0.0;
}

/// Derivative of the ReLU activation function.
/// (d/dx)ReLU(x) = { 1, x > 0
///                 { 0, x <= 0
pub fn reluDerivative(x: f64) f64 {
    return if (x > 0.0) 1.0 else 0.0;
}

//
//
//                           *** unit tests ***
//
//
//

test "Weights and biases initialized" {
    var net = Network.init();
    net.initializeWeights();
    net.initializeBiases();

    var rng = std.rand.DefaultPrng.init(@abs(std.time.timestamp()));
    var idx: usize = rng.random().intRangeAtMost(usize, 0, net.weights1.len);
    if (idx >= net.weights1.len) {
        idx = @mod(idx, net.weights1.len);
    }
    std.debug.print("\n\nWeight at index {}: {}\n\n", .{ idx, net.weights1[idx] });
    if (idx >= net.biases1.len) {
        idx = @mod(idx, net.biases1.len);
    }
    std.debug.print("Bias at index {}: {}\n\n", .{ idx, net.biases1[idx] });
}

test "Epoch" {
    const start = std.time.milliTimestamp();

    // TODO: investigate differences between page_allocator and GeneralPurposeAllocator.
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const image_file = "/home/maa/dev/source/ml/mnist/nn-in-zig/raw/train-images-idx3-ubyte";
    const label_file = "/home/maa/dev/source/ml/mnist/nn-in-zig/raw/train-labels-idx1-ubyte";

    var num_images: usize = 0;
    var num_labels: usize = 0;

    const images = try mnist_loader.loadImages(allocator, image_file, &num_images);
    const labels = try mnist_loader.loadLabels(allocator, label_file, &num_labels);
    try std.testing.expectEqual(num_images, num_labels);

    var net = Network.init();
    net.initializeWeights();
    net.initializeBiases();

    var loss: f64 = 0.0;
    for (0..num_images) |i| {
        var input = [_]f64{0} ** Network.image_size;
        for (0..Network.image_size) |j| {
            const image = images[i][j];
            input[j] = @as(f64, @floatFromInt(image)) / 255.0;
        }
        net.forward(&input, 0.0, true);
        net.backpropagate(&input, labels[i], 1e-3);
        loss += net.cost(labels[i]);
    }
    loss /= @as(f64, @floatFromInt(num_images));

    std.debug.print("\n\nLoss: {d}\n\n", .{loss});
    std.debug.print("Runtime: {} ms\n\n", .{std.time.milliTimestamp() - start});
}
