import json
import random
import re
import speech_recognition as sr
import pyttsx3
import webbrowser
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

intents = data["intents"]


patterns = []
tags = []
responses_map = {}

for intent in intents:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])
    responses_map[intent["tag"]] = intent["responses"]


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text


cleaned_patterns = [clean(p) for p in patterns]


model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

pattern_embeddings = model.encode(
    cleaned_patterns,
    show_progress_bar=False,
    convert_to_numpy=True
)


recognizer = sr.Recognizer()
engine = pyttsx3.init()


def listen():
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)

        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except:
        return ""


def speak(text):
    print("Bot:", text)
    engine.say(text)
    engine.runAndWait()


fallbacks = [
    "I'm not sure I understood that.",
    "Can you rephrase it?",
    "I'm still learning, try again.",
    "Sorry, I didn't get that."
]


def get_time():
    now = datetime.now()
    return f"The time is {now.strftime('%H:%M')}"


def open_apps(user_input):
    user_input = user_input.lower()

    if "google" in user_input:
        webbrowser.open("https://www.google.com")
        return "Opening Google"

    if "youtube" in user_input:
        webbrowser.open("https://www.youtube.com")
        return "Opening YouTube"

    if "weather" in user_input:
        webbrowser.open("https://www.google.com/search?q=weather")
        return "Opening weather"

    return None


def is_stop(text):
    text = text.lower()
    return any(word in text for word in ["stop", "exit", "quit"])


def get_response(user_input):
    user_input = clean(user_input)

    user_embedding = model.encode(
        [user_input],
        show_progress_bar=False,
        convert_to_numpy=True
    )

    similarity = cosine_similarity(user_embedding, pattern_embeddings)

    best_index = similarity.argmax()
    best_score = similarity[0][best_index]

    if best_score < 0.35:
        return random.choice(fallbacks)

    elif best_score < 0.5:
        tag = tags[best_index]
        return "I'm not fully sure, but " + random.choice(responses_map[tag])

    else:
        tag = tags[best_index]
        return random.choice(responses_map[tag])


def main():
    speak("AI Assistant started.")

    while True:
        user_input = listen()

        if not user_input:
            continue

        if is_stop(user_input):
            speak("Have a nice day")
            return

        if "time" in user_input:
            speak(get_time())
            continue

        app_response = open_apps(user_input)
        if app_response:
            speak(app_response)
            continue

        response = get_response(user_input)
        speak(response)


if __name__ == "__main__":
    main()