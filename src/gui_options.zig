const std = @import("std");
const c = @cImport({
    @cInclude("gui_options.h");
});

pub const GuiOptions = extern struct {
    load_model: c_int,
    epochs: c_int,
    learning_rate: f32,
    draw_digit: c_int,
    confirmed: c_int,
    digit_image: [28 * 28]u8,
};

extern fn gui_get_user_options(opts: *GuiOptions) c_int;

pub fn getUserOptions() !GuiOptions {
    var opts: GuiOptions = undefined;
    if (gui_get_user_options(&opts) != 0) return error.GuiFailed;
    return opts;
}
