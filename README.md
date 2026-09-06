# 🤖 Hugging Face Chatbot

A complete working chatbot using Hugging Face transformers with multiple implementations:
1. **Web-based chatbot** (Flask + HTML/CSS/JavaScript)
2. **Advanced terminal chatbot** (Intent recognition, Sentiment analysis)
3. **Simple chatbot** (Quick start version)

## Features

✨ **Sentiment Analysis** - Understand user emotions  
🎯 **Intent Recognition** - Detect what user wants  
💬 **Natural Response Generation** - Generate human-like responses  
📊 **Conversation History** - Track chat history  
🌐 **Web Interface** - Beautiful interactive UI  
⚡ **GPU Support** - Faster responses with CUDA  

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/psjadhav2314-rgb/huggingface-chatbot.git
cd huggingface-chatbot
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** First run will download models (~1-2 GB). Be patient!

---

## Usage

### Option 1: Web-Based Chatbot (Recommended)

Run the Flask server:
```bash
python app.py
```

Open browser and go to: **http://localhost:5000**

Features:
- Beautiful web interface
- Real-time sentiment and intent display
- Clear chat history
- Responsive design

### Option 2: Advanced Terminal Chatbot

```bash
python advanced_chatbot.py
```

Features:
- Intent recognition
- Sentiment analysis
- Knowledge base Q&A
- Conversation history (type `history`)
- Exit with `quit`

Example:
```
You: Hello!
Bot: Hey there! How's it going?
[Analysis]
  Intent: greeting (confidence: 0.98)
  Sentiment: POSITIVE (score: 0.92)
```

### Option 3: Simple Terminal Chatbot (Quick Start)

```bash
python simple_chatbot.py
```

Minimal version for quick testing:
```
You: How are you?
Bot: I'm doing great, thanks for asking!
[Sentiment: POSITIVE]
```

---

## Architecture

### Components

```
User Input
    ↓
Sentiment Analysis (emotional context)
    ↓
Intent Recognition (what user wants)
    ↓
Response Generation (create reply)
    ↓
Output Display
```

### Models Used

| Model | Task | Purpose |
|-------|------|---------|
| **gpt2** | Text Generation | Generate bot responses |
| **distilbert** | Sentiment Analysis | Detect user emotions |
| **roberta** | Intent Classification | Understand user intent |

---

## API Endpoints (Web Version)

### POST `/api/chat`
Send message and get response
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

Response:
```json
{
  "success": true,
  "user_message": "Hello!",
  "bot_response": "Hi! How can I help?",
  "sentiment": "POSITIVE",
  "sentiment_score": 0.95,
  "intent": "greeting"
}
```

### GET `/api/history`
Get conversation history
```bash
curl http://localhost:5000/api/history
```

### POST `/api/clear`
Clear conversation history
```bash
curl -X POST http://localhost:5000/api/clear
```

---

## Customization

### Add Custom Intents
Edit `app.py` or `advanced_chatbot.py`:
```python
intents = ["greeting", "question", "complaint", "custom_intent"]
```

### Add Knowledge Base
In `advanced_chatbot.py`:
```python
self.knowledge_base = {
    "your_key": "Your answer here",
    "another_key": "Another answer"
}
```

### Change Model
```python
# Use a better model
generator = pipeline("text-generation", model="mistral-7b")

# Or use conversation-specific model
conversational = pipeline("conversational", model="microsoft/DialoGPT-medium")
```

---

## Examples

### Example 1: Sentiment Detection
```python
from app import chatbot

result = chatbot.chat("I'm very frustrated!")
print(result)
# Output:
# {
#   "response": "...",
#   "sentiment": "NEGATIVE",
#   "sentiment_score": 0.98,
#   "intent": "complaint"
# }
```

### Example 2: Custom Intent Handling
```python
def handle_intent(intent, user_input):
    if intent == "complaint":
        return "I sincerely apologize for your experience."
    elif intent == "greeting":
        return "Hello! Welcome!"
    else:
        return generate_response(user_input)
```

---

## Performance Tips

🚀 **Use GPU** - Responses are 5-10x faster with CUDA
```python
# Automatically uses GPU if available
device = 0 if torch.cuda.is_available() else -1
```

📦 **Use Smaller Models** - For faster responses
```python
# Instead of gpt2 (124M parameters)
generator = pipeline("text-generation", model="distilgpt2")  # 82M parameters
```

⚡ **Cache Models** - Download once, use many times

---

## Troubleshooting

### Model Download Issues
```bash
# Clear cache
rm -rf ~/.cache/huggingface/

# Set custom cache dir
export HF_HOME=/path/to/cache
python app.py
```

### Out of Memory
Use smaller models:
```python
model = pipeline("text-generation", model="gpt2")  # Use distilgpt2 instead
```

### Slow Responses
- Ensure GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Use smaller models
- Increase max_length carefully (trade-off with speed)

---

## Project Structure

```
huggingface-chatbot/
├── app.py                    # Main Flask web app
├── simple_chatbot.py         # Terminal chatbot (simple)
├── advanced_chatbot.py       # Terminal chatbot (advanced)
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Web UI
└── README.md                # This file
```

---

## Next Steps

1. **Deploy to cloud** - Use Heroku, AWS, or Google Cloud
2. **Improve responses** - Fine-tune on custom data
3. **Add database** - Store conversations in PostgreSQL/MongoDB
4. **Integrate with platforms** - Connect to Telegram, Discord, Slack
5. **Add voice** - Use speech-to-text and text-to-speech

---

## Resources

📚 **Hugging Face Documentation**
- https://huggingface.co/docs/transformers/

📚 **Hugging Face Course**
- https://huggingface.co/course/

🎓 **Flask Documentation**
- https://flask.palletsprojects.com/

🔗 **Model Hub**
- https://huggingface.co/models

---

## License

MIT License - Feel free to use and modify!

---

## Author

Created by: psjadhav2314-rgb

## Support

For issues and questions, create an issue on GitHub!

Happy chatting! 🤖💬
