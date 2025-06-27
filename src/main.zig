const std = @import("std");
const Network = @import("nn.zig").Network;
const mnist_loader = @import("mnist_loader.zig");
const gui = @import("gui_options.zig");
const assert = std.debug.assert;

pub fn main() anyerror!void {
    const allocator = std.heap.page_allocator;

    // Loop the entire program flow, returning to GUI after each run
    while (true) {
        const opts: gui.GuiOptions = try gui.getUserOptions();
        if (opts.confirmed == 0) {
            std.debug.print("GUI canceled or closed, exiting.\n", .{});
            break;
        }
        std.debug.print("GUI Options Selected - load_model: {}, epochs: {}, learning_rate: {d}, draw_digit: {}\n", .{ opts.load_model, opts.epochs, opts.learning_rate, opts.draw_digit });

        const epochs: usize = if (opts.epochs > 0) @as(usize, @intCast(opts.epochs)) else 10;
        var learning_rate: f64 = if (opts.learning_rate > 0.0) @as(f64, opts.learning_rate) else 1.6e-3;
        const dropout_rate: f64 = 0.0;

        const train_image_file = "raw/train-images-idx3-ubyte";
        const train_label_file = "raw/train-labels-idx1-ubyte";
        const test_image_file = "raw/t10k-images-idx3-ubyte";
        const test_label_file = "raw/t10k-labels-idx1-ubyte";

        var train_num_images: usize = 0;
        var train_num_labels: usize = 0;
        var test_num_images: usize = 0;
        var test_num_labels: usize = 0;

        const train_images = try mnist_loader.loadImages(allocator, train_image_file, &train_num_images);
        const train_labels = try mnist_loader.loadLabels(allocator, train_label_file, &train_num_labels);
        const test_images = try mnist_loader.loadImages(allocator, test_image_file, &test_num_images);
        const test_labels = try mnist_loader.loadLabels(allocator, test_label_file, &test_num_labels);

        if (train_num_images != train_num_labels or test_num_images != test_num_labels) {
            std.debug.print("Mismatch between images and labels.\n", .{});
            return;
        }

        var net = Network.init();
        if (opts.load_model == 1) {
            try net.load("mnist_model.bin");
        } else {
            net.initializeWeights();
            net.initializeBiases();
            std.debug.print("Training samples: {}\nTesting samples: {}\n\n", .{ train_num_images, test_num_images });
            std.debug.print("Running {} epochs...\n", .{epochs});
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
                        std.debug.print("Epoch {}: Sample {}/{}\r", .{ epoch + 1, i + 1, train_num_images });
                    }
                }
                epoch_loss /= @as(f64, @floatFromInt(train_num_images));
                std.debug.print("\n\t Loss: {d:.4}\n", .{epoch_loss});
                std.debug.print("\t Learning rate: {d:.6}\n", .{learning_rate});
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
        std.debug.print("\nTest Accuracy: {d}%\n", .{accuracy});

        // User drawing and classification
        if (opts.draw_digit == 1) {
            var input = [_]f64{0} ** Network.image_size;
            for (0..Network.image_size) |j| {
                input[j] = @as(f64, @floatFromInt(opts.digit_image[j])) / 255.0;
            }
            net.forward(&input, 0.0, false);
            const prediction = net.predict();
            std.debug.print("Predicted digit: {}\n", .{prediction});
        }
    }
}
