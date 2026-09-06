from flask import Flask, request, jsonify, render_template
from transformers import pipeline, Conversation
import torch

app = Flask(__name__)

# Initialize Hugging Face pipelines
print("Loading Hugging Face models... (this may take a moment)")
try:
    # Text generation for responses
    generator = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
    
    # Sentiment analysis
    sentiment_analyzer = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
    
    # Intent recognition (zero-shot classification)
    intent_classifier = pipeline("zero-shot-classification", device=0 if torch.cuda.is_available() else -1)
    
    print("✓ Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    generator = None
    sentiment_analyzer = None
    intent_classifier = None

# Store conversation history
conversation_history = []

class Chatbot:
    def __init__(self):
        self.conversation_history = []
    
    def analyze_sentiment(self, text):
        """Analyze user sentiment"""
        try:
            result = sentiment_analyzer(text)[0]
            return result
        except Exception as e:
            return {"label": "NEUTRAL", "score": 0.0}
    
    def detect_intent(self, text):
        """Detect user intent"""
        intents = ["greeting", "question", "complaint", "request", "goodbye"]
        try:
            result = intent_classifier(text, intents)
            return result['labels'][0]
        except Exception as e:
            return "question"
    
    def generate_response(self, user_input):
        """Generate bot response"""
        try:
            # Detect sentiment and intent
            sentiment = self.analyze_sentiment(user_input)
            intent = self.detect_intent(user_input)
            
            # Create context-aware prompt
            if sentiment['label'] == 'NEGATIVE':
                prompt = f"The user seems upset. Respond helpfully: {user_input}. Response:"
            elif intent == 'greeting':
                prompt = f"User greeted. Respond warmly: {user_input}. Response:"
            elif intent == 'goodbye':
                prompt = f"User is saying goodbye. Respond politely: {user_input}. Response:"
            else:
                prompt = f"User: {user_input}\nBot:"
            
            # Generate response
            response = generator(prompt, max_length=80, num_return_sequences=1, do_sample=True)
            generated_text = response[0]['generated_text']
            
            # Extract only the bot response part
            if "Response:" in generated_text:
                bot_response = generated_text.split("Response:")[-1].strip()
            elif "Bot:" in generated_text:
                bot_response = generated_text.split("Bot:")[-1].strip()
            else:
                bot_response = generated_text.replace(prompt, "").strip()
            
            return {
                "response": bot_response,
                "sentiment": sentiment['label'],
                "sentiment_score": round(sentiment['score'], 2),
                "intent": intent
            }
        except Exception as e:
            return {
                "response": "I'm having trouble understanding. Could you rephrase?",
                "sentiment": "NEUTRAL",
                "sentiment_score": 0.0,
                "intent": "question",
                "error": str(e)
            }
    
    def chat(self, user_input):
        """Main chat function"""
        # Add to history
        self.conversation_history.append({"role": "user", "message": user_input})
        
        # Generate response
        bot_output = self.generate_response(user_input)
        
        # Add to history
        self.conversation_history.append({
            "role": "bot",
            "message": bot_output["response"],
            "sentiment": bot_output.get("sentiment"),
            "intent": bot_output.get("intent")
        })
        
        return bot_output

# Initialize chatbot
chatbot = Chatbot()

# Routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({"error": "Empty message"}), 400
        
        # Get bot response
        bot_response = chatbot.chat(user_input)
        
        return jsonify({
            "success": True,
            "user_message": user_input,
            "bot_response": bot_response["response"],
            "sentiment": bot_response["sentiment"],
            "sentiment_score": bot_response["sentiment_score"],
            "intent": bot_response["intent"]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    return jsonify({"history": chatbot.conversation_history})

@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    chatbot.conversation_history = []
    return jsonify({"success": True, "message": "History cleared"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
