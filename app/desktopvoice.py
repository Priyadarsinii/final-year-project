import tkinter as tk
import requests
import json
import pyttsx3

class GestureRecognitionApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Gesture Recognition")
        
        # Create a label to display the gesture prediction
        self.prediction_label = tk.Label(self.window, text="", font=("Arial", 50))
        self.prediction_label.pack(pady=50)
        
        # Create a button to get the gesture prediction
        self.get_prediction_button = tk.Button(self.window, text="Get Prediction", command=self.get_prediction)
        self.get_prediction_button.pack(pady=10)
        
        # Initialize the text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)
        
        # Run the app
        self.window.mainloop()
        
    def get_prediction(self):
        # Send a GET request to the Flask server and get the JSON response
        response = requests.get("http://localhost:5000/")
        json_response = json.loads(response.content)
        
        # Get the gesture prediction from the JSON response
        prediction = json_response["prediction"]
        
        # Update the prediction label with the new prediction
        self.prediction_label.configure(text=prediction)
        
        # Speak the gesture prediction
        self.engine.say(prediction)
        self.engine.runAndWait()
        
if __name__ == "__main__":
    app = GestureRecognitionApp()
