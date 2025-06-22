const std = @import("std");
const Network = @import("nn.zig").Network;
const mnist_loader = @import("mnist_loader.zig");
const assert = std.debug.assert;

pub fn main() anyerror!void {
    const stdout = std.io.getStdOut().writer();
    const allocator = std.heap.page_allocator;

    const epochs: usize = 10;
    var learning_rate: f64 = 1.6e-3;
    const dropout_rate: f64 = 0.0;

    const train_image_file = "raw/train-images-idx3-ubyte";
    const train_label_file = "raw/train-labels-idx1-ubyte";
    const test_image_file = "raw/t10k-images-idx3-ubyte";
    const test_label_file = "raw/t10k-labels-idx1-ubyte";

    // Load MNIST data
    var train_num_images: usize = 0;
    var train_num_labels: usize = 0;

    var test_num_images: usize = 0;
    var test_num_labels: usize = 0;

    const train_images = try mnist_loader.loadImages(allocator, train_image_file, &train_num_images);
    const train_labels = try mnist_loader.loadLabels(allocator, train_label_file, &train_num_labels);
    const test_images = try mnist_loader.loadImages(allocator, test_image_file, &test_num_images);
    const test_labels = try mnist_loader.loadLabels(allocator, test_label_file, &test_num_labels);

    if (train_num_images != train_num_labels or
        test_num_images != test_num_labels)
    {
        try stdout.print("Mismatch between images and labels.\n", .{});
        return;
    }

    var net = Network.init();

    var load = false;
    try stdout.print("Load network from mnist_model.bin? [y/N]\n", .{});
    const l = std.io.getStdIn().reader().readByte();

    if (try l == 'y') {
        load = true;
        try net.load("mnist_model.bin");
        // Consume the newline character left in the input buffer
        _ = try std.io.getStdIn().reader().readByte();
    }

    const start = std.time.milliTimestamp();

    if (!load) {
        net.initializeWeights();
        net.initializeBiases();

        try stdout.print("Training samples: {}\nTesting samples: {}\n\n", .{ train_num_images, test_num_images });
        try stdout.print("Running {} epochs...\n", .{epochs});

        // training loop
        for (0..epochs) |epoch| {
            var epoch_loss: f64 = 0.0;

            for (0..train_num_images) |i| {
                var input = [_]f64{0} ** Network.image_size;
                for (0..Network.image_size) |j| {
                    const image = train_images[i][j];
                    input[j] = @as(f64, @floatFromInt(image)) / 255.0;
                }
                net.forward(&input, dropout_rate, true);
                net.backpropagate(&input, train_labels[i], learning_rate);
                epoch_loss += net.cost(train_labels[i]);
                if ((i + 1) % 1000 == 0 or i == train_num_images - 1) {
                    try stdout.print("Epoch {}: Sample {}/{}\r", .{ epoch + 1, i + 1, train_num_images });
                }
            }
            epoch_loss /= @as(f64, @floatFromInt(train_num_images));

            try stdout.print("\n\t Loss: {d:.4}\n", .{epoch_loss});
            try stdout.print("\t Learning rate: {d:.6}\n", .{learning_rate});
            learning_rate *= 0.9;
        }

        try net.save("mnist_model.bin");
    }

    // testing loop
    var correct: usize = 0;
    for (0..test_num_images) |i| {
        var input = [_]f64{0} ** Network.image_size;

        for (0..Network.image_size) |j| {
            const image = test_images[i][j];
            input[j] = @as(f64, @floatFromInt(image)) / 255.0;
        }

        net.forward(&input, dropout_rate, false);
        if (net.predict() == @as(usize, test_labels[i])) {
            correct += 1;
        }
    }

    const accuracy = 100.0 * @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(test_num_images));
    try stdout.print("\nTest Accuracy: {d}%\n", .{accuracy});

    const stop = std.time.milliTimestamp();
    const elapsed_ms: f64 = @as(f64, @floatFromInt(stop - start));
    if (!load)
        try stdout.print("Runtime: {d} seconds.\n", .{elapsed_ms / 1000});

    // User drawing and classification
    try stdout.print("Draw a digit? [Y/n]\n", .{});
    const n = std.io.getStdIn().reader().readByte();

    if (try n != 'n' and try n != 'N') {
        const user_image = try mnist_loader.getUserImage(allocator);
        var input = [_]f64{0} ** Network.image_size;
        for (0..Network.image_size) |j| {
            input[j] = @as(f64, @floatFromInt(user_image[j])) / 255.0;
        }
        net.forward(&input, 0.0, false);
        const prediction = net.predict();
        try stdout.print("Predicted digit: {}\n", .{prediction});
    }
}
