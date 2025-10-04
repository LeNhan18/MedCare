from flask import Flask, request, jsonify
from src.chatbot.conversation_handler import ConversationHandler

app = Flask(__name__)
conversation_handler = ConversationHandler()

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    response = conversation_handler.handle_message(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)