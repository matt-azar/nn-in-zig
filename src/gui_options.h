#ifndef GUI_OPTIONS_H
#define GUI_OPTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int load_model;      // 1 = load, 0 = train new
    int epochs;
    float learning_rate;
    int draw_digit;      // 1 = yes, 0 = no
    int confirmed;       // 1 = OK (Enter pressed), 0 = closed or ESC
    unsigned char digit_image[28*28]; // filled if draw_digit==1
} GuiOptions;

int gui_get_user_options(GuiOptions* out_opts);

#ifdef __cplusplus
}
#endif

#endif // GUI_OPTIONS_H
