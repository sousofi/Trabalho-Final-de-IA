import pygame
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import random
import time
import joblib
import copy
import os 
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from PIL import ImageOps

# Carregar os pesos do modelo
model = joblib.load('mlp_model.pkl')
scaler = joblib.load('scaler.pkl')

img_dir_men_teste = "Dataset/Test/Male"
img_dir_women_teste = "Dataset/Test/Female"

filenames_men_teste = [os.path.join(img_dir_men_teste, file) for file in os.listdir(img_dir_men_teste)[:7000]]
filenames_women_teste = [os.path.join(img_dir_women_teste, file) for file in os.listdir(img_dir_women_teste)[:7000]]
filenames_teste = filenames_men_teste + filenames_women_teste

images_teste = [Image.open(file).resize((128, 128)) for file in filenames_teste]

labels_men_teste = [0 for _ in filenames_men_teste]
labels_women_teste = [1 for _ in filenames_women_teste]
labels_teste = labels_men_teste + labels_women_teste
labels_teste = np.array(labels_teste)

classes = {0: "Male", 1: "Female"}

def predict_gender(image):
    image_aux = copy.deepcopy(image)
    image_gray = image_aux.convert('L')
    image_array = np.array(image_gray).reshape(1, -1)  
    images_teste_normalized = scaler.transform(image_array)
    prediction = model.predict(images_teste_normalized)[0]
    return prediction

def load_specific_gender_images(images, labels, class_name, num_images=12):
    class_label = 0 if class_name == "Male" else 1
    selected_images, selected_labels = [], []

    filtered_images = [img for img, label in zip(images, labels) if label == class_label]

    # Selecionar aleatoriamente num_images imagens
    selected_indices = random.sample(range(len(filtered_images)), min(num_images, len(filtered_images)))

    selected_images = [filtered_images[i] for i in selected_indices]
    selected_labels = [class_label] * len(selected_images)
            
    return selected_images, selected_labels

images_female, labels_female = load_specific_gender_images(images_teste, labels_teste, "Female", num_images=12)
images_male, labels_male = load_specific_gender_images(images_teste, labels_teste, "Male", num_images=12)

images = images_female + images_male
labels = labels_female + labels_male
cards = list(zip(images, labels))
random.shuffle(cards)

# Inicializa o Pygame
pygame.init()

# Configurações da tela inicial e da tela principal
start_screen_width = 400
start_screen_height = 250
game_screen_width = 1050
game_screen_height = 900

screen = pygame.display.set_mode((start_screen_width, start_screen_height))
pygame.display.set_caption("Jogo da Memória")

# Cores
popup_background_color = (240, 240, 240)
popup_border_color = (0, 0, 0)
popup_font = pygame.font.Font(None, 36)
background_color = (240, 240, 240)
window_color = (255, 250, 250)
button_color = (255, 192, 203)
text_color = (0, 0, 0)
grid_color = (250, 250, 250)
card_color = (177, 52, 235)  # Cor do cartão

# Configurações do texto
font_path = os.path.join("Anton", "Anton-Regular.ttf")
font = pygame.font.Font(font_path, 52)
title_text = font.render("Jogo da Memória", True, text_color)

# Configurações do botão
button_width = 150
button_height = 70
button_x = (start_screen_width - button_width) // 2
button_y = (start_screen_height - button_height) // 2 + 50

# Função para desenhar o botão
def draw_button():
    pygame.draw.rect(screen, button_color, (button_x, button_y, button_width, button_height), 0, border_radius=25)
    font_button = pygame.font.Font(None, 48)
    text = font_button.render("Jogar", True, text_color)
    text_width = text.get_width()
    text_height = text.get_height()
    text_x = button_x + (button_width - text_width) // 2  # Centraliza horizontalmente
    text_y = button_y + (button_height - text_height) // 2  # Centraliza verticalmente

    # Desenha o texto centralizado no botão
    screen.blit(text, (text_x, text_y))

def draw_board_grid():
    grid_size = 50
    for x in range(0, game_screen_width, grid_size):
        for y in range(0,game_screen_height, grid_size):
            pygame.draw.rect(screen, grid_color, pygame.Rect(x, y, grid_size, grid_size), 1)

# Função para desenhar a tela de início
def draw_start_screen():
    screen.fill(background_color)
    grid_size = 50
    for x in range(0, start_screen_width, grid_size):
        for y in range(0, start_screen_height, grid_size):
            pygame.draw.rect(screen, grid_color, pygame.Rect(x, y, grid_size, grid_size), 1)

    title_x = (start_screen_width - title_text.get_width()) // 2  # Centraliza horizontalmente
    title_y = 50  # Posição vertical do título (ajuste conforme necessário)

    screen.blit(title_text, (title_x, title_y))

    draw_button()
    pygame.display.flip()

def draw_rounded_rect(surface, color, rect, corner_radius):
    shape_surface = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
    pygame.draw.rect(shape_surface, color, shape_surface.get_rect(), border_radius=corner_radius)
    
    # Certifique-se de criar um pygame.Rect a partir do tuple
    shape_rect = pygame.Rect(rect)  # Crie um pygame.Rect a partir do tuple (x, y, width, height)
    
    surface.blit(shape_surface, shape_rect.topleft)  # Use shape_rect.topleft

def draw_shadow(surface, rect, offset, corner_radius):
    shadow_rect = rect.move(offset, offset)
    shadow_surface = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
    shadow_color = (0, 0, 0, 80)
    pygame.draw.rect(shadow_surface, shadow_color, shadow_surface.get_rect(), border_radius=corner_radius)
    surface.blit(shadow_surface, shadow_rect.topleft)

def draw_rounded_image(surface, image, rect, corner_radius):
    # Converter imagem PIL para array NumPy
    image = np.array(image)
    if len(image.shape) == 2:  # Imagem em escala de cinza
        # Converte a imagem em cinza para uma imagem RGB triplicando as camadas
        image = np.stack([image] * 3, axis=-1)
    
    # Corrige a orientação da imagem
    image = np.flipud(image)  # Inverte a imagem verticalmente
    image = np.rot90(image, 2)
    
    # Redimensionar a imagem para o tamanho do retângulo
    image = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    image = pygame.transform.scale(image, rect.size)
    
    # Criar uma superfície para a imagem arredondada
    rounded_image_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
    rounded_image_surface.fill((0, 0, 0, 0))

    # Desenhar a imagem na superfície arredondada
    rounded_image_surface.blit(image, (0, 0))
    
    # Criar uma máscara para o arredondamento
    mask_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(mask_surface, (255, 255, 255, 255), mask_surface.get_rect(), border_radius=corner_radius)

    # Aplicar a máscara para arredondar a imagem
    rounded_image_surface.blit(mask_surface, (0, 0), None, pygame.BLEND_RGBA_MIN)
    surface.blit(rounded_image_surface, rect.topleft)

def draw_grid(turned_cards):
    card_width = 120
    card_height = 170
    grid_rows = 4
    grid_cols = 6
    padding = 40
    start_x = (game_screen_width - (grid_cols * card_width + (grid_cols - 1) * padding)) // 2
    start_y = (game_screen_height - (grid_rows * card_height + (grid_rows - 1) * padding)) // 2

    draw_board_grid()
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = start_x + col * (card_width + padding)
            y = start_y + row * (card_height + padding)
            card_rect = pygame.Rect(x, y, card_width, card_height)
            if (row, col) in turned_cards:
                image = cards[turned_cards[(row, col)]][0]
                draw_shadow(screen, card_rect, 10, 20)
                draw_rounded_image(screen, image, card_rect, 20)
            else:
                draw_shadow(screen, card_rect, 10, 20)
                draw_rounded_rect(screen, card_color, card_rect, 20)

def show_popup(message):
    popup_font = pygame.font.Font(None, 36)
    popup_surface = popup_font.render(message, True, text_color)
    popup_rect = popup_surface.get_rect(center=(game_screen_width // 2, game_screen_height // 2))

    # Aumentar o tamanho da caixinha do pop-up
    popup_width = popup_rect.width + 60
    popup_height = popup_rect.height + 60
    popup_background = pygame.Surface((popup_width, popup_height), pygame.SRCALPHA)

    # Definir cores para o fundo e borda do pop-up
    popup_background_color = (255, 255, 255, 230)  # Fundo branco com leve transparência
    popup_border_color = (0, 0, 0)  # Cor da borda preta

    # Desenhar o fundo do pop-up com bordas arredondadas
    pygame.draw.rect(popup_background, popup_background_color, pygame.Rect(0, 0, popup_width, popup_height), border_radius=20)
    pygame.draw.rect(popup_background, popup_border_color, pygame.Rect(0, 0, popup_width, popup_height), 2, border_radius=20)

    # Efeito de fade-in
    for alpha in range(0, 256, 10):
        popup_background.set_alpha(alpha)
        screen.blit(popup_background, (popup_rect.x - 30, popup_rect.y - 30))
        screen.blit(popup_surface, popup_rect)
        pygame.display.flip()
        pygame.time.delay(5)

    # Tempo de exibição do pop-up
    time.sleep(0.8)
    
def show_game_over():
    screen.fill((0, 0, 0))  
    game_over_font = pygame.font.Font(None, 120)
    game_over_text = game_over_font.render("Game Over", True, (255, 0, 0))  
    game_over_rect = game_over_text.get_rect(center=(game_screen_width // 2, game_screen_height // 2))
    screen.blit(game_over_text, game_over_rect)
    pygame.display.flip()
    time.sleep(3)  
    pygame.quit()  
    exit()  

def show_you_won():
    screen.fill((0, 0, 0))  
    you_won_font = pygame.font.Font(None, 120)
    you_won_text = you_won_font.render("You won!", True, (255, 255, 255))  
    you_won_rect = you_won_text.get_rect(center=(game_screen_width // 2, game_screen_height // 2))
    screen.blit(you_won_text, you_won_rect)
    pygame.display.flip()
    time.sleep(3)  
    pygame.quit()  
    exit()  

turned_cards = {}
matched_cards = set()

recently_shown = []

def main_game():
    global recently_shown
    screen = pygame.display.set_mode((game_screen_width, game_screen_height))  # Muda para a tela maior
    pygame.display.set_caption("Jogo da Memória - Principal")
    
    global turned_cards, matched_cards
    running = True
    card_genders = [int(predict_gender(img)) for img, _ in cards]

    while running:
        pair = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if len(matched_cards) < len(cards):
            available_indices = [i for i in range(len(cards)) if i not in matched_cards]
            
            available_indices = [i for i in available_indices if i not in recently_shown]

            if len(available_indices) < 2:
                recently_shown = []
                available_indices = [i for i in range(len(cards)) if i not in matched_cards]

            card_indices = random.sample(available_indices, 2)
            card1, card2 = card_indices[0], card_indices[1]

            recently_shown.extend([card1, card2])
            if len(recently_shown) > 2: 
                recently_shown.pop(0)

            turned_cards[(card1 // 6, card1 % 6)] = card1
            turned_cards[(card2 // 6, card2 % 6)] = card2

            screen.fill(background_color)
            draw_grid(turned_cards)
            pygame.display.flip()
            
            time.sleep(1)

            actual_label1 = cards[card1][1]
            actual_label2 = cards[card2][1]  

            if card_genders[card1] == card_genders[card2]:
                if actual_label1 == actual_label2:
                    pair = True
                    pair_gender = classes[card_genders[card1]]
                    if pair_gender == 'Male':
                        print(f"Encontrou um par: Homem")
                        matched_cards.update({card1, card2})
                        show_popup(f"Encontrou um par: Homem")
                    else:
                        print(f"Encontrou um par: Mulher")
                        matched_cards.update({card1, card2})
                        show_popup(f"Encontrou um par: Mulher")
                else:
                    print("Erro de classificação! Jogo Terminado.")
                    show_popup(f"Encontrou um par: {classes[card_genders[card1]]}")
                    time.sleep(2)
                    show_game_over()
            else:
                print("Não é um par!")
                turned_cards.pop((card1 // 6, card1 % 6))
                turned_cards.pop((card2 // 6, card2 % 6))

            screen.fill(background_color)
            draw_grid(turned_cards)
            pygame.display.flip()
            time.sleep(0.5)
        else:
            print("Todas as cartas foram combinadas. Você venceu!")
            show_you_won()
    pygame.quit()  # Garante que o Pygame seja encerrado
    exit() 

# Loop principal que controla as telas
def start_game_loop():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                if button_x <= mouse_x <= button_x + button_width and button_y <= mouse_y <= button_y + button_height:
                    main_game()  # Se o botão for clicado, inicia o jogo principal

        draw_start_screen()
        pygame.display.flip()

    pygame.quit()
    exit()

# Inicia o loop do jogo
start_game_loop()
