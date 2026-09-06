"""
Simple standalone chatbot (no web server required)
Run this directly in terminal for quick testing
"""

from transformers import pipeline
import torch

class SimpleChatbot:
    def __init__(self):
        print("Loading models...")
        self.generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
        self.sentiment = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
        print("✓ Ready to chat!\n")
        self.history = []
    
    def chat(self, user_input):
        # Analyze sentiment
        sentiment = self.sentiment(user_input)[0]
        sentiment_label = sentiment['label']
        
        # Add to history
        self.history.append(f"You: {user_input}")
        
        # Create context-aware prompt
        if sentiment_label == 'NEGATIVE':
            prompt = f"User seems upset. Be empathetic: {user_input}\nBot:"
        else:
            prompt = f"User: {user_input}\nBot:"
        
        # Generate response
        response = self.generator(prompt, max_length=80, do_sample=True)[0]['generated_text']
        
        # Extract bot response
        bot_response = response.split("Bot:")[-1].strip()
        
        # Add to history
        self.history.append(f"Bot: {bot_response}")
        
        return bot_response, sentiment_label

def main():
    bot = SimpleChatbot()
    
    print("=" * 50)
    print("Simple Hugging Face Chatbot")
    print("=" * 50)
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        response, sentiment = bot.chat(user_input)
        print(f"Bot: {response}")
        print(f"[Sentiment: {sentiment}]\n")

if __name__ == "__main__":
    main()
