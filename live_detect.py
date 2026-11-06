import cv2
import pygame
import time
from model import Model

def main():
    # Define which labels from settings.yaml should trigger the alarm
    VIOLENT_LABELS = [
        'fight on a street',
        'fire on a street',
        'street violence',
        'car crash',
        'violence in office',
        'fire in office'
    ]
    
    # Initialize pygame mixer for audio
    pygame.mixer.init()
    try:
        alarm_sound = pygame.mixer.Sound("warning-alarm.wav")
    except FileNotFoundError:
        print("Error: warning-alarm.wav not found.")
        return

    model = Model()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Live detection started. Alarm will stop 2 seconds after violence stops.")

    # --- NEW: Variable to track when violence was last seen ---
    last_violence_time = 0 
    STOP_DELAY = 2.0  # Seconds to wait before stopping audio

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB as required by the CLIP model for accuracy
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get prediction from the model
        prediction = model.predict(image=rgb_frame)
        label = prediction['label']
        confidence = prediction['confidence']
        
        current_time = time.time()

        # LOGIC: Audio Control with Delay
        if label in VIOLENT_LABELS:
            # Update the last time we saw violence
            last_violence_time = current_time
            
            # Play sound if not already playing
            if not pygame.mixer.get_busy():
                alarm_sound.play(-1)  # -1 loops the audio continuously
                print(f"!!! ALERT: {label.upper()} !!!")
        else:
            # If violence is NOT detected, check the timer
            # Only stop if it has been more than STOP_DELAY seconds since the last violence
            if current_time - last_violence_time > STOP_DELAY:
                alarm_sound.stop()

        # UI Feedback
        display_text = f"{label.title()} ({confidence:.2f})"
        color = (0, 0, 255) if label in VIOLENT_LABELS else (0, 255, 0)
        
        cv2.putText(frame, display_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Violence Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources
    alarm_sound.stop()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == '__main__':
    main()