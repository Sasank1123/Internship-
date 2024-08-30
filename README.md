import random
import nltk
from transformers import pipeline

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('punkt')

# Initialize AI/ML components
classifier = pipeline("sentiment-analysis")
qa_pipeline = pipeline("question-answering")

# Function to greet the user
def greet_user():
    greetings = ["Hello! How can I help you today?", 
                 "Hi there! What can I do for you?", 
                 "Greetings! How may I assist you?"]
    return random.choice(greetings)

# Function to handle user input
def chatbot(query):
    if any(word in query.lower() for word in ["hello", "hi", "hey", "greetings"]):
        return random.choice(["Hello!", "Hi there!", "Greetings!", "Hey!", "Hi!"])
    
    if "sentiment" in query.lower():
        sentiment = classifier(query)[0]['label']
        return "You seem to be in a good mood!" if sentiment == 'POSITIVE' else "I'm sorry to hear that."

    if "synonym" in query.lower():
        word = query.split()[-1]
        synonyms = [syn.lemmas()[0].name() for syn in nltk.corpus.wordnet.synsets(word)]
        return f"Synonyms of '{word}' are: {', '.join(synonyms[:5])}" if synonyms else "I couldn't find any synonyms."

    if "?" in query:
        answer = qa_pipeline(question=query, context="I am a chatbot designed to help you.")
        return answer['answer']

    return "Sorry, I didn't understand that. Could you please rephrase?"

# Main loop
if __name__ == "__main__":
    print(greet_user())
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        print("Bot:", chatbot(user_input))
