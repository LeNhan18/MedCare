const apiUrl = 'http://localhost:5000/api/chat'; // Update with your API endpoint

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const userInput = input.value.trim();
        if (userInput) {
            appendMessage('You: ' + userInput);
            input.value = '';
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userInput }),
            });

            if (response.ok) {
                const data = await response.json();
                appendMessage('Bot: ' + data.reply);
            } else {
                appendMessage('Bot: Sorry, I could not process your request.');
            }
        }
    });

    function appendMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.textContent = message;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the bottom
    }
});