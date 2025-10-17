"""
Flask API cho Medical Chatbot
Cung cấp endpoints để tương tác với chatbot y tế
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import yaml
import logging

# Thêm đường dẫn để import các modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from medical_pipeline import MedicalChatbotPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Không thể import pipeline: {e}")
    PIPELINE_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    API_CONFIG = config.get('api', {})
except Exception as e:
    print(f"⚠️ Không thể load config: {e}")
    API_CONFIG = {'host': '0.0.0.0', 'port': 5000}

# Initialize pipeline
pipeline = None
if PIPELINE_AVAILABLE:
    try:
        pipeline = MedicalChatbotPipeline(config_path)
        print("🚀 Medical Chatbot Pipeline đã được khởi tạo!")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo pipeline: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    """Health check endpoint"""
    pipeline_status = None
    if pipeline:
        pipeline_status = pipeline.get_pipeline_status()
    
    return jsonify({
        "message": "Medical Chatbot API is running!",
        "version": "1.0.0",
        "pipeline_available": PIPELINE_AVAILABLE,
        "pipeline_status": pipeline_status,
        "endpoints": [
            "GET / - Health check",
            "POST /chat - Chat with medical bot",
            "POST /drug-info - Get drug information",  
            "POST /risk-check - Check drug risks",
            "GET /status - Get system status"
        ]
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        message = data.get('message', '').strip()
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        user_profile = data.get('user_profile', {})
        
        logger.info(f"Received message: {message}")
        
        # Process with pipeline if available
        if pipeline:
            try:
                result = pipeline.process_message(message, user_profile)
                
                response = {
                    "success": True,
                    "message": message,
                    "response": result['final_response'],
                    "intent": result.get('intent', {}),
                    "entities": result.get('entities', {}),
                    "drug_recommendations": result.get('drug_recommendations', []),
                    "risk_assessments": result.get('risk_assessments', []),
                    "processing_steps": len(result.get('processing_steps', [])),
                    "warnings": result.get('warnings', [])
                }
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                return jsonify({
                    "success": False,
                    "error": "Pipeline processing error",
                    "message": message,
                    "response": "Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu của bạn. Vui lòng thử lại."
                }), 500
        
        else:
            # Fallback response when pipeline is not available
            response = {
                "success": True,
                "message": message,
                "response": _generate_fallback_response(message),
                "intent": {"intent": "unknown", "confidence": 0.5},
                "entities": {"entities": []},
                "drug_recommendations": [],
                "risk_assessments": [],
                "warnings": ["Hệ thống đang ở chế độ hạn chế. Vui lòng tham khảo ý kiến bác sĩ."]
            }
            
            return jsonify(response)
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": "Đã có lỗi xảy ra. Vui lòng thử lại."
        }), 500

@app.route('/drug-info', methods=['POST'])
def drug_info():
    """Get detailed drug information"""
    try:
        data = request.get_json()
        drug_name = data.get('drug_name', '').strip()
        
        if not drug_name:
            return jsonify({"error": "Drug name is required"}), 400
        
        if pipeline and pipeline.drug_recommender:
            drug_info = pipeline.drug_recommender.get_drug_info(drug_name)
            
            if drug_info:
                return jsonify({
                    "success": True,
                    "drug_name": drug_name,
                    "drug_info": drug_info
                })
            else:
                return jsonify({
                    "success": False,
                    "drug_name": drug_name,
                    "message": f"Không tìm thấy thông tin về thuốc {drug_name}"
                }), 404
        else:
            return jsonify({
                "success": False,
                "error": "Drug information service not available"
            }), 503
    
    except Exception as e:
        logger.error(f"Drug info error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/risk-check', methods=['POST'])
def risk_check():
    """Check drug risks for specific user profile"""
    try:
        data = request.get_json()
        drug_name = data.get('drug_name', '').strip()
        user_profile = data.get('user_profile', {})
        
        if not drug_name:
            return jsonify({"error": "Drug name is required"}), 400
        
        if not user_profile:
            return jsonify({"error": "User profile is required"}), 400
        
        if pipeline and pipeline.risk_checker:
            risk_result = pipeline.risk_checker.check_risk_rule_based(drug_name, user_profile)
            
            return jsonify({
                "success": True,
                "drug_name": drug_name,
                "user_profile": user_profile,
                "risk_assessment": risk_result
            })
        else:
            return jsonify({
                "success": False,
                "error": "Risk checking service not available"
            }), 503
    
    except Exception as e:
        logger.error(f"Risk check error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/status')
def status():
    """Get system status"""
    pipeline_status = None
    if pipeline:
        pipeline_status = pipeline.get_pipeline_status()
    
    return jsonify({
        "api_status": "running",
        "pipeline_available": PIPELINE_AVAILABLE,
        "pipeline_status": pipeline_status,
        "config": {
            "host": API_CONFIG.get('host', '0.0.0.0'),
            "port": API_CONFIG.get('port', 5000)
        }
    })

def _generate_fallback_response(message):
    """Generate fallback response when pipeline is not available"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['đau đầu', 'sốt', 'ho', 'nghẹt']):
        return """Tôi hiểu bạn đang có một số triệu chứng. 
        
🏥 **Khuyến nghị:**
- Đối với đau đầu, sốt: có thể dùng paracetamol theo hướng dẫn
- Đối với ho, nghẹt mũi: nghỉ ngơi, uống nhiều nước
- Nếu triệu chứng kéo dài >3 ngày hoặc nặng hơn, hãy đến gặp bác sĩ

⚠️ **Lưu ý:** Đây chỉ là thông tin tham khảo. Vui lòng tham khảo ý kiến chuyên gia y tế."""
    
    elif any(word in message_lower for word in ['thuốc', 'paracetamol', 'ibuprofen']):
        return """Để có thông tin chính xác về thuốc, bạn nên:

📋 **Tham khảo:**
- Đọc kỹ hướng dẫn sử dụng trên bao bì
- Hỏi ý kiến dược sĩ tại nhà thuốc
- Tham khảo bác sĩ nếu có bệnh nền

⚠️ **An toàn:** Không tự ý dùng thuốc khi không rõ tác dụng phụ."""
    
    else:
        return """Xin chào! Tôi là chatbot y tế, có thể giúp bạn:

🤖 **Chức năng:**
- Tư vấn về các triệu chứng thường gặp
- Thông tin về thuốc không kê đơn (OTC)
- Khuyến nghị sơ cứu cơ bản

💬 **Cách sử dụng:** Hãy mô tả triệu chứng hoặc hỏi về thuốc bạn quan tâm.

⚠️ **Lưu ý:** Thông tin chỉ mang tính tham khảo, không thay thế ý kiến bác sĩ."""

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    host = API_CONFIG.get('host', '0.0.0.0')
    port = API_CONFIG.get('port', 5000)
    debug = os.getenv('FLASK_ENV') == 'development'
    
    print(f"🌐 Starting Medical Chatbot API on {host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"🤖 Pipeline available: {PIPELINE_AVAILABLE}")
    
    app.run(debug=debug, host=host, port=port)