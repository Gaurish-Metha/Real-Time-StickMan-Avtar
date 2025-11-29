import cv2
import pygame
import sys
import os
import numpy as np

# Add the directory containing the script to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tracker import HolisticTracker
from audio import AudioProcessor
from avatar import Avatar

def draw_gradient_loading(screen, width, height):
    """Draws a simple gradient and loading text."""
    # Draw Gradient (Blue to Black)
    for y in range(height):
        # Interpolate color
        c = int(20 * (1 - y/height)) # Dark fade
        color = (10, 10 + int(40 * (y/height)), 20 + int(60 * (y/height)))
        pygame.draw.line(screen, color, (0, y), (width, y))
    
    # Draw Text
    font = pygame.font.SysFont("Arial", 40, bold=True)
    text = font.render("Launching Gaurish Realtime Avatar...", True, (0, 255, 255))
    text_rect = text.get_rect(center=(width//2, height//2))
    screen.blit(text, text_rect)
    
    # Subtext
    font_small = pygame.font.SysFont("Arial", 20)
    text_small = font_small.render("Initializing AI Modules...", True, (150, 150, 150))
    text_small_rect = text_small.get_rect(center=(width//2, height//2 + 50))
    screen.blit(text_small, text_small_rect)
    
    pygame.display.flip()

def main():
    pygame.init()
    
    # Settings
    WIDTH, HEIGHT = 1280, 720
    flags = pygame.HWSURFACE | pygame.DOUBLEBUF
    screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
    pygame.display.set_caption("Gaurish Realtime Avatar")
    clock = pygame.time.Clock()

    # --- SHOW LOADING SCREEN ---
    draw_gradient_loading(screen, WIDTH, HEIGHT)
    
    # Allow the event loop to pump once so the window appears
    pygame.event.pump()

    print("Initializing Tracker... (High Accuracy Mode)")
    tracker = HolisticTracker()
    
    print("Initializing Audio...")
    audio = AudioProcessor()
    audio.start()
    
    avatar = Avatar(WIDTH, HEIGHT)
    
    print("Opening Camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    print("System Ready. Press ESC to exit.")
    
    show_camera = True
    # Preview geometry
    preview_w = 320
    preview_h = 180
    button_rect = pygame.Rect(WIDTH - 110, 10, 100, 30)
    font_ui = pygame.font.SysFont("Arial", 16)

    running = True
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_v: # Keyboard shortcut
                    show_camera = not show_camera
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    if button_rect.collidepoint(event.pos):
                        show_camera = not show_camera

        # Video Capture
        ret, frame = cap.read()
        if not ret:
            print("Camera frame lost.")
            continue
            
        frame = cv2.flip(frame, 1)
        
        # Tracking (Optimized input)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        results = tracker.process(small_frame)
        data = tracker.extract_landmarks(results)
        
        vol = audio.get_volume()
        
        # Render
        screen.fill((10, 10, 15)) # Deep dark background
        
        # Grid floor
        for i in range(0, WIDTH, 100):
             pygame.draw.line(screen, (30, 30, 40), (i, HEIGHT), (WIDTH/2, HEIGHT/2 - 50), 1)
        pygame.draw.line(screen, (0, 255, 255), (0, HEIGHT - 50), (WIDTH, HEIGHT - 50), 2)
        
        avatar.update_and_draw(screen, data, vol)
        
        # --- Webcam Preview (FIXED) ---
        if show_camera:
            # 1. Resize
            preview_img = cv2.resize(frame, (preview_w, preview_h))
            # 2. Convert BGR to RGB
            preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            # 3. Transpose (Swap axes to match Pygame's (width, height, colors))
            # OpenCV is (Height, Width, Colors), Pygame surface expects (Width, Height, Colors)
            preview_img = np.transpose(preview_img, (1, 0, 2))
            # 4. Make Surface
            preview_surf = pygame.surfarray.make_surface(preview_img)
            
            screen.blit(preview_surf, (WIDTH - preview_w - 10, 50))
            
            # Draw border around preview
            pygame.draw.rect(screen, (0, 255, 255), (WIDTH - preview_w - 10, 50, preview_w, preview_h), 2)

        # --- UI Button ---
        # Draw Button Background
        btn_color = (0, 200, 100) if show_camera else (200, 50, 50)
        pygame.draw.rect(screen, btn_color, button_rect, 0, 5)
        pygame.draw.rect(screen, (255, 255, 255), button_rect, 2, 5)
        
        # Draw Button Text
        btn_txt = "Cam: ON" if show_camera else "Cam: OFF"
        txt_surf = font_ui.render(btn_txt, True, (255, 255, 255))
        # Center text
        txt_rect = txt_surf.get_rect(center=button_rect.center)
        screen.blit(txt_surf, txt_rect)

        # Debug Info (FPS)
        fps = int(clock.get_fps())
        font = pygame.font.SysFont("Arial", 18)
        fps_text = font.render(f"FPS: {fps}", True, (100, 100, 100))
        screen.blit(fps_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(60)

    # Cleanup
    print("Shutting down...")
    cap.release()
    audio.stop()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
