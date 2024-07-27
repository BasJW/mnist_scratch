"""
- This code lets us draw a number on a 28*28 grid
- We then use the NN to classify the number we have drawn
"""

import pygame
import sys
import math
from main import classify

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 280*3 + 100, 280*3
GRID_SIZE = 28
CELL_SIZE = (WIDTH - 100) // GRID_SIZE
BUTTON_WIDTH, BUTTON_HEIGHT = 80, 40

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MNIST Drawing Tool")

# Create a numpy array to store the drawing
global drawing_array
drawing_array = [[0 for i in range(GRID_SIZE)] for j in range(GRID_SIZE)]

font = pygame.font.Font(None, 24)

def draw_grid():
    for x in range(0, WIDTH - 100, CELL_SIZE):
        pygame.draw.line(screen, (50, 50, 50), (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, (50, 50, 50), (0, y), (WIDTH - 100, y))

def get_cell(pos):
    x, y = pos
    return x // CELL_SIZE, y // CELL_SIZE

def draw_brush(cell, radius=2, intensity=1.0):
    cx, cy = cell
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                distance = math.sqrt(dx**2 + dy**2)
                if distance <= radius:
                    drawing_array[y][x] = min(drawing_array[y][x] + intensity * (1 - distance/radius), 1)

def update_display():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            color = int(drawing_array[y][x] * 255)
            pygame.draw.rect(screen, (color, color, color), 
                             (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    draw_grid()
    draw_buttons()
    pygame.display.flip()

def draw_buttons():
    button_y = (HEIGHT - 2*BUTTON_HEIGHT - 10) // 2
    # Clear button
    pygame.draw.rect(screen, GRAY, (WIDTH - 90, button_y, BUTTON_WIDTH, BUTTON_HEIGHT))
    clear_text = font.render("Clear", True, BLACK)
    screen.blit(clear_text, (WIDTH - 75, button_y + 10))
    
    # Classify button
    pygame.draw.rect(screen, GRAY, (WIDTH - 90, button_y + BUTTON_HEIGHT + 10, BUTTON_WIDTH, BUTTON_HEIGHT))
    classify_text = font.render("Classify", True, BLACK)
    screen.blit(classify_text, (WIDTH - 85, button_y + BUTTON_HEIGHT + 20))

def clear_canvas():
    global drawing_array
    drawing_array = [[0 for i in range(GRID_SIZE)] for j in range(GRID_SIZE)]


def show_popup(result):
    popup_width, popup_height = 200, 100
    popup_x = (WIDTH - popup_width) // 2
    popup_y = (HEIGHT - popup_height) // 2
    
    popup_surface = pygame.Surface((popup_width, popup_height))
    popup_surface.fill(WHITE)
    pygame.draw.rect(popup_surface, BLACK, (0, 0, popup_width, popup_height), 2)
    
    result_text = font.render(f"Classified as: {result}", True, BLACK)
    text_rect = result_text.get_rect(center=(popup_width//2, popup_height//2))
    popup_surface.blit(result_text, text_rect)
    
    screen.blit(popup_surface, (popup_x, popup_y))
    pygame.display.flip()
    
    waiting_for_click = True
    while waiting_for_click:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
                waiting_for_click = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                x, y = event.pos
                if x < WIDTH - 100:
                    draw_brush(get_cell(event.pos))
                else:
                    button_y = (HEIGHT - 2*BUTTON_HEIGHT - 10) // 2
                    if WIDTH - 90 <= x <= WIDTH - 10:
                        if button_y <= y <= button_y + BUTTON_HEIGHT:  
                            clear_canvas()
                        elif button_y + BUTTON_HEIGHT + 10 <= y <= button_y + 2*BUTTON_HEIGHT + 10:  
                            classification_result = classify(drawing_array)
                            show_popup(classification_result)
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0] and event.pos[0] < WIDTH - 100:  
                draw_brush(get_cell(event.pos))
        elif event.type == pygame.KEYDOWN:
            pass
    
    screen.fill(BLACK)
    update_display()

pygame.quit()
sys.exit()

