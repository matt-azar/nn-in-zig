/// Wrapper for the C function `draw_and_get_image` that draws a GUI and returns an image.
const std = @import("std");

/// Calls the C function `draw_and_get_image` to draw a digit and retrieve the image data.
/// * @param out_image Pointer to a buffer to store the drawn image.
/// * @return 0 on success, non-zero on error.
/// * @throws error.DrawingGuiFailed if the drawing operation fails.
/// This function is defined in the C file `draw_gui.c`.
extern fn draw_and_get_image(out_image: [*]u8) c_int;

/// Draws a digit on the screen and retrieves the image data.
/// * @param allocator Memory allocator to use for image allocation.
/// * @return A slice of u8 containing the drawn image data.
/// * @throws error.DrawingGuiFailed if the drawing operation fails.
pub fn drawAndGetImage(allocator: std.mem.Allocator) ![]u8 {
    const image = try allocator.alloc(u8, 28 * 28);
    const rc = draw_and_get_image(image.ptr);
    if (rc != 0) return error.DrawingGuiFailed;
    return image;
}
