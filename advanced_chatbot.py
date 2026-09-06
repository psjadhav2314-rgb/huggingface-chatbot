"""
Advanced chatbot with multiple capabilities:
- Intent recognition
- Sentiment analysis
- Question answering
- Context awareness
"""

from transformers import pipeline
import torch

class AdvancedChatbot:
    def __init__(self):
        print("Loading advanced models...")
        device = 0 if torch.cuda.is_available() else -1
        
        self.generator = pipeline("text-generation", model="gpt2", device=device)
        self.sentiment = pipeline("sentiment-analysis", device=device)
        self.intent_classifier = pipeline("zero-shot-classification", device=device)
        self.qa_pipeline = pipeline("question-answering", device=device)
        
        print("✓ All models loaded!\n")
        
        self.conversation_history = []
        self.knowledge_base = {
            "name": "I'm an AI assistant powered by Hugging Face",
            "created": "I was created using transformers library",
            "purpose": "I'm here to have helpful conversations and assist you",
            "ai": "AI stands for Artificial Intelligence",
            "hugging face": "Hugging Face is a company providing NLP models and tools"
        }
    
    def detect_intent(self, text):
        """Detect user's intent"""
        intents = ["greeting", "question", "complaint", "request", "small_talk", "goodbye"]
        result = self.intent_classifier(text, intents)
        return result['labels'][0], result['scores'][0]
    
    def analyze_sentiment(self, text):
        """Analyze sentiment"""
        result = self.sentiment(text)[0]
        return result['label'], result['score']
    
    def answer_question(self, question, context):
        """Answer questions about knowledge base"""
        try:
            result = self.qa_pipeline(question=question, context=context)
            return result['answer']
        except:
            return None
    
    def get_knowledge_response(self, user_input):
        """Check if question is in knowledge base"""
        user_input_lower = user_input.lower()
        
        for key, value in self.knowledge_base.items():
            if key in user_input_lower:
                return value
        
        return None
    
    def generate_response(self, user_input, intent, sentiment):
        """Generate context-aware response"""
        # Check knowledge base first
        knowledge_response = self.get_knowledge_response(user_input)
        if knowledge_response:
            return knowledge_response
        
        # Create context-aware prompt based on intent and sentiment
        prompts = {
            "greeting": f"User greeted. Respond warmly: {user_input}\nBot:",
            "goodbye": f"User is leaving. Say goodbye politely: {user_input}\nBot:",
            "complaint": f"User has a complaint. Be empathetic and helpful: {user_input}\nBot:",
            "question": f"Answer the question: {user_input}\nBot:",
            "small_talk": f"Have a casual conversation: {user_input}\nBot:",
            "request": f"Help with this request: {user_input}\nBot:"
        }
        
        prompt = prompts.get(intent, f"User: {user_input}\nBot:")
        
        # Adjust for negative sentiment
        if sentiment == 'NEGATIVE':
            prompt = f"User seems upset. Be very helpful and empathetic: {user_input}\nBot:"
        
        # Generate response
        response = self.generator(prompt, max_length=100, do_sample=True)[0]['generated_text']
        bot_response = response.split("Bot:")[-1].strip()
        
        return bot_response
    
    def chat(self, user_input):
        """Main chat method"""
        # Analyze input
        intent, intent_score = self.detect_intent(user_input)
        sentiment, sentiment_score = self.analyze_sentiment(user_input)
        
        # Generate response
        response = self.generate_response(user_input, intent, sentiment)
        
        # Store in history
        self.conversation_history.append({
            "user": user_input,
            "bot": response,
            "intent": intent,
            "sentiment": sentiment
        })
        
        return {
            "response": response,
            "intent": intent,
            "intent_score": round(intent_score, 2),
            "sentiment": sentiment,
            "sentiment_score": round(sentiment_score, 2)
        }

def main():
    bot = AdvancedChatbot()
    
    print("=" * 60)
    print("Advanced Hugging Face Chatbot")
    print("=" * 60)
    print("Features: Intent Recognition, Sentiment Analysis, Q&A")
    print("Type 'quit' to exit, 'history' to see chat history\n")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'history':
            print("\n" + "=" * 60)
            print("CONVERSATION HISTORY")
            print("=" * 60)
            for i, item in enumerate(bot.conversation_history, 1):
                print(f"\n[Turn {i}]")
                print(f"You: {item['user']}")
                print(f"Bot: {item['bot']}")
                print(f"Intent: {item['intent']} | Sentiment: {item['sentiment']}")
            print("\n" + "=" * 60)
            continue
        
        # Get response
        result = bot.chat(user_input)
        
        # Display response with analysis
        print(f"\nBot: {result['response']}")
        print(f"\n[Analysis]")
        print(f"  Intent: {result['intent']} (confidence: {result['intent_score']})")
        print(f"  Sentiment: {result['sentiment']} (score: {result['sentiment_score']})")

if __name__ == "__main__":
    main()
