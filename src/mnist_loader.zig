const std = @import("std");

pub fn readInt(file: []const u8) !u32 {
    if (file.len < 4) return error.InvalidFileLength;

    return @as(u32, file[0]) << 24 | @as(u32, file[1]) << 16 | @as(u32, file[2]) << 8 | @as(u32, file[3]);
}

pub fn loadImages(allocator: std.mem.Allocator, file_name: []const u8, num_images: *usize) ![][]u8 {
    // Be careful with relative paths. If you want to be able to open files during tests,
    // you need to have the /raw folder in your /src directory. When you build the project
    // with zig build, it will look for the /raw folder in the project's main directory.
    // The workaround to this is to have /raw in /nn-in-zig and in /nn-in-zig/src.

    // const file = try std.fs.openFileAbsolute(file_name, .{}); // uses absolute path
    const file = try std.fs.cwd().openFile(file_name, .{}); // uses relative path
    defer file.close();

    const file_size: usize = try file.getEndPos();
    const buffer = try file.reader().readAllAlloc(allocator, file_size);
    const magic_number = try readInt(buffer[0..4]);
    if (magic_number != 0x00000803) {
        return error.InvalidMagicNumber;
    }

    num_images.* = try readInt(buffer[4..8]);
    const rows = try readInt(buffer[8..12]);
    const cols = try readInt(buffer[12..16]);
    if (rows != 28 or cols != 28) {
        return error.InvalidDimensions;
    }

    const image_data_start = 16;
    const image_size = rows * cols;
    var images = try allocator.alloc([]u8, num_images.*);
    for (0..num_images.*) |i| {
        images[i] = try allocator.alloc(u8, image_size);
        std.mem.copyForwards(u8, images[i], buffer[image_data_start + i * image_size .. image_data_start + (i + 1) * image_size]);
    }

    return images;
}

pub fn loadLabels(allocator: std.mem.Allocator, file_name: []const u8, num_labels: *usize) ![]u8 {
    // const file = try std.fs.openFileAbsolute(file_name, .{});
    const file = try std.fs.cwd().openFile(file_name, .{});
    defer file.close();

    const file_size: usize = try file.getEndPos();
    const buffer = try file.reader().readAllAlloc(allocator, file_size);
    const magic_number = try readInt(buffer[0..4]);
    if (magic_number != 0x00000801) {
        return error.InvalidMagicNumber;
    }
    num_labels.* = try readInt(buffer[4..8]);

    return buffer[8..];
}

/// Prompt the user to draw a digit in a GUI window.
pub fn getUserImage(allocator: std.mem.Allocator) ![]u8 {
    return try @import("draw_gui.zig").drawAndGetImage(allocator);
}

pub fn displayImage(image: []const u8) void {
    for (0..28) |i| {
        for (0..28) |j| {
            const pixel = image[i * 28 + j];
            if (pixel > 64 and pixel < 128) {
                std.debug.print("{s}", .{"."});
            } else if (pixel >= 128 and pixel < 192) {
                std.debug.print("{s}", .{"*"});
            } else {
                std.debug.print("{s}", .{if (pixel > 128) "#" else " "});
            }
        }
        std.debug.print("\n", .{});
    }
}

//  .     .   .      o       .          .       *  . .     .
//    .  *  |     .    .==============  .     .   .     * .    .
//        --o--         | unit tests |     *    |      ..    .
//     *    |       *  .==============   .    --*--  .     * .
//  .     .    .    .   . . .      .        .   |   .    .  .

// use `zig test mnist_loader.zig` in CLI if you want the program to find a relative path
test "display image" {
    const allocator = std.heap.page_allocator;

    const image_file = "raw/train-images-idx3-ubyte";
    const label_file = "raw/train-labels-idx1-ubyte";

    var num_images: usize = 0;
    var num_labels: usize = 0;

    const images = try loadImages(allocator, image_file, &num_images);
    const labels = try loadLabels(allocator, label_file, &num_labels);

    try std.testing.expectEqual(num_images, num_labels);

    std.debug.print("\n\nNumber of images: {}\n\n", .{num_images});
    if (num_images > 0) {
        var r = std.Random.DefaultPrng.init(@abs(std.time.timestamp()));
        const idx: usize = r.random().intRangeAtMost(usize, 0, num_images);
        std.debug.print("Displaying image {}:\n", .{idx});
        displayImage(images[idx]);
        std.debug.print("Label: {}\n\n", .{labels[idx]});
    } else {
        std.debug.print("No images loaded.\n", .{});
    }
}

// test "get user image" {
//     const allocator = std.heap.page_allocator;

//     const image = try getUserImage(allocator);
//     defer allocator.free(image);

//     std.debug.print("User drawn image:\n", .{});
//     displayImage(image);
// }
