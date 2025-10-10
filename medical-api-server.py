#!/usr/bin/env python3
"""
Medical Chatbot API Backend
Hỗ trợ N8N Workflow với các endpoint tìm kiếm thuốc
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import json
from datetime import datetime
import re
from fuzzywuzzy import fuzz, process

app = Flask(__name__)
CORS(app)

# Load medical dataset
print("🏥 Loading medical dataset...")
try:
    df_drugs = pd.read_csv('data/medical_dataset_training.csv')
    print(f"✅ Loaded {len(df_drugs)} drugs successfully")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    df_drugs = pd.DataFrame()

class MedicalSearchEngine:
    def __init__(self, dataframe):
        self.df = dataframe
        self.symptom_mapping = self._build_symptom_mapping()
    
    def _build_symptom_mapping(self):
        """Xây dựng mapping từ triệu chứng tiếng Việt"""
        mapping = {}
        if not self.df.empty:
            for idx, row in self.df.iterrows():
                condition_vi = str(row.get('medical_condition_vi', '')).lower()
                condition_en = str(row.get('medical_condition', '')).lower()
                
                # Tạo các từ khóa tìm kiếm
                keywords = [condition_vi, condition_en]
                
                # Thêm từ khóa phổ biến
                if 'đau đầu' in condition_vi or 'headache' in condition_en:
                    keywords.extend(['đau đầu', 'nhức đầu', 'migraine'])
                elif 'mụn' in condition_vi or 'acne' in condition_en:
                    keywords.extend(['mụn', 'mụn trứng cá', 'acne'])
                elif 'giảm cân' in condition_vi or 'weight loss' in condition_en:
                    keywords.extend(['giảm cân', 'béo phì', 'diet'])
                
                for keyword in keywords:
                    if keyword and len(keyword) > 2:
                        if keyword not in mapping:
                            mapping[keyword] = []
                        mapping[keyword].append(idx)
        
        return mapping
    
    def search_drugs(self, symptoms, severity='mild', language='vi', limit=5):
        """Tìm kiếm thuốc dựa trên triệu chứng"""
        if self.df.empty:
            return {'drugs': [], 'total': 0, 'message': 'Database not available'}
        
        results = []
        seen_drugs = set()
        
        for symptom in symptoms:
            symptom_clean = symptom.lower().replace('_', ' ')
            
            # Tìm kiếm trực tiếp
            matches = self.df[
                (self.df['medical_condition_vi'].str.contains(symptom_clean, case=False, na=False)) |
                (self.df['medical_condition'].str.contains(symptom_clean, case=False, na=False))
            ]
            
            # Tìm kiếm fuzzy nếu không có kết quả trực tiếp
            if matches.empty:
                all_conditions = (
                    self.df['medical_condition_vi'].dropna().astype(str).tolist() + 
                    self.df['medical_condition'].dropna().astype(str).tolist()
                )
                
                fuzzy_matches = process.extract(symptom_clean, all_conditions, limit=3, scorer=fuzz.partial_ratio)
                fuzzy_conditions = [match[0] for match in fuzzy_matches if match[1] > 60]
                
                for condition in fuzzy_conditions:
                    fuzzy_results = self.df[
                        (self.df['medical_condition_vi'].str.contains(condition, case=False, na=False)) |
                        (self.df['medical_condition'].str.contains(condition, case=False, na=False))
                    ]
                    matches = pd.concat([matches, fuzzy_results])
            
            # Thêm kết quả vào danh sách
            for _, drug in matches.iterrows():
                drug_name = drug.get('drug_name', 'Unknown')
                if drug_name not in seen_drugs:
                    seen_drugs.add(drug_name)
                    
                    drug_info = {
                        'drug_name': drug_name,
                        'medical_condition': drug.get('medical_condition', ''),
                        'condition_vi': drug.get('medical_condition_vi', ''),
                        'side_effects': drug.get('side_effects', ''),
                        'side_effects_vi': drug.get('side_effects_vi', ''),
                        'drug_classes': drug.get('drug_classes', ''),
                        'drug_class_vi': drug.get('drug_classes_vi', ''),
                        'rx_otc': drug.get('rx_otc', ''),
                        'rating': drug.get('rating', 0),
                        'description_vi': drug.get('medical_condition_description_vi', ''),
                        'brand_names': drug.get('brand_names', ''),
                        'severity_match': severity
                    }
                    
                    results.append(drug_info)
        
        # Sắp xếp theo rating và ưu tiên OTC cho severity nhẹ
        results.sort(key=lambda x: (
            -float(x['rating']) if x['rating'] and str(x['rating']) != 'nan' else 0,
            x['rx_otc'] == 'OTC' if severity == 'mild' else x['rx_otc'] == 'Rx'
        ), reverse=True)
        
        return {
            'drugs': results[:limit],
            'total': len(results),
            'message': f'Found {len(results)} drugs for symptoms: {", ".join(symptoms)}'
        }

# Initialize search engine
search_engine = MedicalSearchEngine(df_drugs)

@app.route('/')
def home():
    """Trang chủ API"""
    html = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Chatbot API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
            .method { background: #e74c3c; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; margin-right: 10px; }
            .get { background: #27ae60; }
            .post { background: #e74c3c; }
            pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .stats { display: flex; justify-content: space-around; margin: 20px 0; }
            .stat { text-align: center; background: #3498db; color: white; padding: 15px; border-radius: 5px; flex: 1; margin: 0 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Medical Chatbot API</h1>
            
            <div class="stats">
                <div class="stat">
                    <h3>{{ total_drugs }}</h3>
                    <p>Tổng số thuốc</p>
                </div>
                <div class="stat">
                    <h3>{{ vietnamese_conditions }}</h3>
                    <p>Triệu chứng tiếng Việt</p>
                </div>
                <div class="stat">
                    <h3>{{ api_version }}</h3>
                    <p>Phiên bản API</p>
                </div>
            </div>
            
            <h2>📋 API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/search-drugs</strong>
                <p>Tìm kiếm thuốc dựa trên triệu chứng</p>
                <pre>{
  "symptoms": ["đau_đầu", "sốt"],
  "severity": "mild",
  "language": "vi"
}</pre>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/drugs</strong>
                <p>Lấy danh sách tất cả thuốc</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/chat</strong>
                <p>Endpoint chính cho N8N Webhook</p>
                <pre>{
  "message": "Tôi bị đau đầu",
  "userId": "user123",
  "sessionId": "session456"
}</pre>
            </div>
            
            <h2>🔧 Cách sử dụng với N8N</h2>
            <ol>
                <li>Import workflow JSON vào N8N</li>
                <li>Cập nhật URL endpoint trong HTTP Request nodes</li>
                <li>Activate workflow</li>
                <li>Gửi POST request đến webhook URL</li>
            </ol>
            
            <h2>📊 Trạng thái hệ thống</h2>
            <ul>
                <li>✅ Database: Đã tải thành công</li>
                <li>✅ Search Engine: Hoạt động</li>
                <li>✅ CORS: Đã bật cho tất cả origins</li>
                <li>✅ JSON API: Sẵn sàng</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    stats = {
        'total_drugs': len(df_drugs) if not df_drugs.empty else 0,
        'vietnamese_conditions': len(df_drugs['medical_condition_vi'].dropna()) if not df_drugs.empty else 0,
        'api_version': 'v1.0'
    }
    
    return render_template_string(html, **stats)

@app.route('/api/search-drugs', methods=['POST'])
def search_drugs():
    """API endpoint để tìm kiếm thuốc cho N8N"""
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        severity = data.get('severity', 'mild')
        language = data.get('language', 'vi')
        limit = data.get('limit', 5)
        
        if not symptoms:
            return jsonify({
                'success': False,
                'message': 'Vui lòng cung cấp triệu chứng',
                'drugs': []
            }), 400
        
        results = search_engine.search_drugs(symptoms, severity, language, limit)
        
        return jsonify({
            'success': True,
            'message': results['message'],
            'drugs': results['drugs'],
            'total': results['total'],
            'query': {
                'symptoms': symptoms,
                'severity': severity,
                'language': language
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi server: {str(e)}',
            'drugs': []
        }), 500

@app.route('/api/drugs', methods=['GET'])
def get_all_drugs():
    """Lấy danh sách tất cả thuốc"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        search = request.args.get('search', '')
        
        filtered_df = df_drugs
        if search:
            filtered_df = df_drugs[
                (df_drugs['drug_name'].str.contains(search, case=False, na=False)) |
                (df_drugs['medical_condition_vi'].str.contains(search, case=False, na=False))
            ]
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        drugs = []
        for _, drug in filtered_df.iloc[start_idx:end_idx].iterrows():
            drugs.append({
                'drug_name': drug.get('drug_name', ''),
                'condition_vi': drug.get('medical_condition_vi', ''),
                'drug_class_vi': drug.get('drug_classes_vi', ''),
                'rx_otc': drug.get('rx_otc', ''),
                'rating': drug.get('rating', 0)
            })
        
        return jsonify({
            'success': True,
            'drugs': drugs,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': len(filtered_df),
                'pages': (len(filtered_df) + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi server: {str(e)}'
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Endpoint chính cho chatbot - tương thích với N8N webhook"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        user_id = data.get('userId', 'anonymous')
        session_id = data.get('sessionId', str(datetime.now().timestamp()))
        
        if not message:
            return jsonify({
                'success': False,
                'message': 'Vui lòng nhập tin nhắn',
                'userId': user_id,
                'sessionId': session_id
            }), 400
        
        # Trả về response để N8N xử lý
        return jsonify({
            'success': True,
            'message': message,
            'userId': user_id,
            'sessionId': session_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'received'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi server: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database_loaded': not df_drugs.empty,
        'total_drugs': len(df_drugs) if not df_drugs.empty else 0
    })

if __name__ == '__main__':
    print("🚀 Starting Medical Chatbot API Server...")
    print("📋 Available endpoints:")
    print("   • GET  / - Trang chủ API")
    print("   • POST /api/search-drugs - Tìm kiếm thuốc")
    print("   • GET  /api/drugs - Danh sách thuốc")
    print("   • POST /api/chat - Chat endpoint")
    print("   • GET  /health - Health check")
    print("\n🌐 Server running on: http://localhost:3001")
    print("📖 Tài liệu API: http://localhost:3001")
    
    app.run(host='0.0.0.0', port=3001, debug=True)