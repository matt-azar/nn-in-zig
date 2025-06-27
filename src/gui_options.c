#include "gui_options.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int draw_and_get_image(unsigned char *out_image);

// Helper to render text
static void render_text(SDL_Renderer *renderer, TTF_Font *font,
                        const char *text, int x, int y) {
    SDL_Color color = {255, 255, 255, 255};
    SDL_Surface *surface = TTF_RenderText_Solid(font, text, color);
    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_Rect dst = {x, y, surface->w, surface->h};
    SDL_RenderCopy(renderer, texture, NULL, &dst);
    SDL_FreeSurface(surface);
    SDL_DestroyTexture(texture);
}

int gui_get_user_options(GuiOptions *out_opts) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        return 1;
    if (TTF_Init() != 0) {
        SDL_Quit();
        return 2;
    }
    SDL_Window *window =
        SDL_CreateWindow("Program Options", SDL_WINDOWPOS_CENTERED,
                         SDL_WINDOWPOS_CENTERED, 800, 400, 0);
    if (!window) {
        TTF_Quit();
        SDL_Quit();
        return 3;
    }
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);
    if (!renderer) {
        SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
        return 4;
    }
    TTF_Font *font =
        TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20);
    if (!font) {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
        return 5;
    }

    int running = 1;
    int load_model = 0;
    int draw_digit = 1;
    int confirmed = 0;
    int epochs = 5;
    float learning_rate = 0.0016f;
    char epoch_str[8] = "5";
    char lr_str[16] = "0.0016";
    int input_field = 0; // 0=none, 1=epochs, 2=lr
    SDL_StartTextInput();
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = 0;
            } else if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                case SDLK_ESCAPE:
                    running = 0;
                    break;
                case SDLK_RETURN:
                    confirmed = 1;
                    running = 0;
                    break;
                case SDLK_TAB:
                    input_field = (input_field == 1) ? 2 : 1;
                    break;
                case SDLK_1:
                    load_model = 1;
                    break;
                case SDLK_2:
                    load_model = 0;
                    break;
                case SDLK_3:
                    draw_digit = 1;
                    break;
                case SDLK_4:
                    draw_digit = 0;
                    break;
                case SDLK_BACKSPACE:
                    if (input_field) {
                        char *target = (input_field == 1) ? epoch_str : lr_str;
                        size_t len = strlen(target);
                        if (len > 0)
                            target[len - 1] = '\0';
                    }
                    break;
                default:
                    break;
                }
            } else if (event.type == SDL_TEXTINPUT && input_field) {
                char *target = (input_field == 1) ? epoch_str : lr_str;
                size_t len = strlen(target);
                size_t max_len = (input_field == 1) ? sizeof(epoch_str) - 1
                                                    : sizeof(lr_str) - 1;
                if (len < max_len)
                    strcat(target, event.text.text);
            }
        }
        SDL_SetRenderDrawColor(renderer, 30, 30, 30, 255);
        SDL_RenderClear(renderer);
        render_text(renderer, font, "[1] Load Model", 30, 30);
        render_text(renderer, font, load_model ? "[X]" : "[ ]", 10, 30);
        render_text(renderer, font, "[2] Train New", 30, 60);
        render_text(renderer, font, !load_model ? "[X]" : "[ ]", 10, 60);
        render_text(renderer, font, "Epochs:", 30, 110);
        // render Epochs value, allow empty
        if (strlen(epoch_str) > 0)
            render_text(renderer, font, epoch_str, 120, 110);
        else
            render_text(renderer, font, " ", 120, 110);
        if (input_field == 1)
            render_text(renderer, font, "<-", 200, 110);
        render_text(renderer, font, "Learning Rate:", 30, 150);
        // render Learning Rate value, allow empty
        if (strlen(lr_str) > 0)
            render_text(renderer, font, lr_str, 180, 150);
        else
            render_text(renderer, font, " ", 180, 150);
        if (input_field == 2)
            render_text(renderer, font, "<-", 300, 150);
        render_text(renderer, font, "[3] Draw Digit", 30, 200);
        render_text(renderer, font, draw_digit ? "[X]" : "[ ]", 10, 200);
        render_text(renderer, font, "[4] Skip Drawing", 30, 230);
        render_text(renderer, font, !draw_digit ? "[X]" : "[ ]", 10, 230);
        render_text(renderer, font, "Tab: Switch Field, Enter: OK", 30, 350);
        // Show key mappings for toggles
        render_text(renderer, font,
                    "1=Load Model, 2=Train New, 3=Draw Digit, 4=Skip Drawing",
                    30, 370);
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }
    // After exiting the GUI loop, log selected options for debugging
    SDL_StopTextInput();
    // printf("[Options selected] load_model=%d, draw_digit=%d, epochs=%s, "
    //        "learning_rate=%s, confirmed=%d\n",
    //        load_model, draw_digit, epoch_str, lr_str, confirmed);
    out_opts->load_model = load_model;
    out_opts->draw_digit = draw_digit;
    out_opts->epochs = atoi(epoch_str);
    out_opts->learning_rate = (float)atof(lr_str);
    out_opts->confirmed = confirmed;
    memset(out_opts->digit_image, 0, 28 * 28);

    // Cleanup options GUI before launching draw GUI
    TTF_CloseFont(font);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();

    // Launch draw GUI if requested
    if (confirmed && draw_digit) {
        if (draw_and_get_image(out_opts->digit_image) != 0) {
            return 6;
        }
    }

    return 0;
}
