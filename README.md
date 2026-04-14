#  AI Voice Assistant

An intelligent voice assistant built using Python that can listen to user commands, understand them using Natural Language Processing, and respond with human-like speech.


##  Features

-  Speech Recognition (convert voice to text)
-  Text-to-Speech responses
-  Smart NLP understanding using Sentence Transformers
-  Semantic similarity using Cosine Similarity
-  Open websites (Google, YouTube, Weather)
-  Tell current time
-  Exit command support (stop / quit / exit)


##  How It Works

The assistant uses:
- Text preprocessing (cleaning input text)
- Sentence embeddings using `SentenceTransformer`
- Cosine similarity to match user input with predefined intents
- Dynamic response selection based on similarity score


##  Technologies Used

- Python
- SpeechRecognition
- pyttsx3
- Sentence Transformers
- Scikit-learn
- JSON (for intents data) 

##  Project Structure
