#include <SDL2/SDL.h>
#include <stdlib.h>
#include <string.h>

#define WIN_SIZE 280
#define IMG_SIZE 28
#define SCALE (WIN_SIZE / IMG_SIZE)

/**
 * @brief Draws a digit on the screen and retrieves the image data.
 *
 * @param out_image Pointer to a buffer to store the drawn image.
 * @return int 0 on success, non-zero on error.
 */
int draw_and_get_image(unsigned char *out_image) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        return 1;

    SDL_Window *window = SDL_CreateWindow(
        "Draw a digit then press <Enter>", SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, WIN_SIZE, WIN_SIZE, 0);
    if (!window) {
        SDL_Quit();
        return 2;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);
    if (!renderer) {
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 3;
    }

    unsigned char *buffer = calloc(WIN_SIZE * WIN_SIZE, 1);
    if (!buffer) {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 4;
    }

    int running = 1, drawing = 0;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = 0;
            if (event.type == SDL_MOUSEBUTTONDOWN)
                drawing = 1;
            if (event.type == SDL_MOUSEBUTTONUP)
                drawing = 0;
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_RETURN)
                    running = 0;
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    free(buffer);
                    SDL_DestroyRenderer(renderer);
                    SDL_DestroyWindow(window);
                    SDL_Quit();
                    return 5;
                }
            }
        }
        if (drawing) {
            int x, y;
            SDL_GetMouseState(&x, &y);
            for (int dy = 0; dy < SCALE; ++dy)
                for (int dx = 0; dx < SCALE; ++dx) {
                    int px = x + dx - SCALE / 2;
                    int py = y + dy - SCALE / 2;
                    if (px >= 0 && px < WIN_SIZE && py >= 0 && py < WIN_SIZE)
                        buffer[py * WIN_SIZE + px] = 255;
                }
        }
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        for (int y = 0; y < WIN_SIZE; ++y)
            for (int x = 0; x < WIN_SIZE; ++x)
                if (buffer[y * WIN_SIZE + x]) {
                    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
                    SDL_Rect rect = {x, y, 1, 1};
                    SDL_RenderFillRect(renderer, &rect);
                }
        SDL_RenderPresent(renderer);
        SDL_Delay(10);
    }
    // Downscale to 28x28
    for (int i = 0; i < IMG_SIZE; ++i)
        for (int j = 0; j < IMG_SIZE; ++j) {
            int sum = 0;
            for (int dy = 0; dy < SCALE; ++dy)
                for (int dx = 0; dx < SCALE; ++dx) {
                    int y = i * SCALE + dy;
                    int x = j * SCALE + dx;
                    sum += buffer[y * WIN_SIZE + x];
                }
            out_image[i * IMG_SIZE + j] = sum / (SCALE * SCALE);
        }
    free(buffer);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
