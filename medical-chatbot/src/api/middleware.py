from flask import request, jsonify
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def log_request_info():
    logging.info(f"Request Method: {request.method}")
    logging.info(f"Request Path: {request.path}")
    logging.info(f"Request Body: {request.get_json()}")

def middleware(app):
    @app.before_request
    def before_request():
        log_request_info()

    @app.errorhandler(Exception)
    def handle_exception(e):
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "An internal error occurred."}), 500