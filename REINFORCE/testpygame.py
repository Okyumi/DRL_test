import pygame
pygame.init()
win = pygame.display.set_mode((400, 400))
pygame.display.set_caption("Test Window")

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

pygame.quit()
