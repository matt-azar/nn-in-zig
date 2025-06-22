/// Wrapper for the C function `draw_and_get_image` that draws a GUI and returns an image.
const std = @import("std");

extern fn draw_and_get_image(out_image: [*]u8) c_int;

pub fn drawAndGetImage(allocator: std.mem.Allocator) ![]u8 {
    const image = try allocator.alloc(u8, 28 * 28);
    const rc = draw_and_get_image(image.ptr);
    if (rc != 0) return error.DrawingGuiFailed;
    return image;
}
