# 🚀 Medical Chatbot API Server

## Khởi chạy API Server

### 1. Cài đặt dependencies:
```powershell
pip install -r requirements_api.txt
```

### 2. Khởi chạy server:
```powershell
python medical-api-server.py
```

Server sẽ chạy tại: http://localhost:3001

## 📋 API Endpoints

### 1. Tìm kiếm thuốc (POST /api/search-drugs)
```json
{
  "symptoms": ["đau_đầu", "sốt"],
  "severity": "mild",
  "language": "vi",
  "limit": 5
}
```

### 2. Chat endpoint (POST /api/chat)
```json
{
  "message": "Tôi bị đau đầu",
  "userId": "user123",
  "sessionId": "session456"
}
```

### 3. Danh sách thuốc (GET /api/drugs)
```
?page=1&per_page=20&search=paracetamol
```

## 🔧 Tích hợp với N8N

1. Import workflow từ file: `n8n-workflows/medical-chatbot-workflow-v2.json`
2. Cập nhật URL trong HTTP Request nodes: `http://localhost:3001`
3. Activate workflow
4. Test bằng webhook URL

## 🏥 Tính năng

- ✅ Tìm kiếm thuốc theo triệu chứng tiếng Việt
- ✅ Fuzzy matching cho triệu chứng không chính xác
- ✅ Ưu tiên thuốc OTC cho triệu chứng nhẹ
- ✅ API RESTful hoàn chỉnh
- ✅ CORS enabled cho web integration
- ✅ Health check endpoint

## 📊 Database

- **Tổng số thuốc**: 2,913 drugs
- **Triệu chứng tiếng Việt**: Đầy đủ translation
- **Định dạng**: CSV + JSON
- **Encoding**: UTF-8

## 🔍 Test API

### Sử dụng curl:
```powershell
# Test search drugs
curl -X POST http://localhost:3001/api/search-drugs -H "Content-Type: application/json" -d '{\"symptoms\": [\"đau_đầu\"], \"severity\": \"mild\"}'

# Test chat
curl -X POST http://localhost:3001/api/chat -H "Content-Type: application/json" -d '{\"message\": \"Tôi bị đau đầu\"}'

# Health check
curl http://localhost:3001/health
```

### Sử dụng PowerShell:
```powershell
# Test search drugs
$body = @{
    symptoms = @("đau_đầu")
    severity = "mild"
    language = "vi"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:3001/api/search-drugs" -Method POST -Body $body -ContentType "application/json"
```

## 🎯 Production Deployment

### Với Gunicorn:
```powershell
gunicorn -w 4 -b 0.0.0.0:3001 medical-api-server:app
```

### Với Docker:
```powershell
# Build image
docker build -t medical-chatbot-api .

# Run container
docker run -p 3001:3001 medical-chatbot-api
```

## 🔗 N8N Workflow URLs

Cập nhật các URL sau trong N8N workflow:

- **Drug Search API**: `http://localhost:3001/api/search-drugs`
- **Chat Endpoint**: `http://localhost:3001/api/chat`
- **Health Check**: `http://localhost:3001/health`

## 🚨 Troubleshooting

### Lỗi import modules:
```powershell
pip install --upgrade pip
pip install -r requirements_api.txt
```

### Port 3001 đã được sử dụng:
```python
# Thay đổi port trong medical-api-server.py
app.run(host='0.0.0.0', port=3002, debug=True)
```

### Database không tải được:
- Kiểm tra file `data/medical_dataset_training.csv` tồn tại
- Đảm bảo encoding UTF-8
- Kiểm tra đường dẫn tệp

## 📈 Monitoring

API server cung cấp các thông tin monitoring:

- **Health Status**: GET /health
- **Database Status**: Số lượng thuốc loaded
- **API Performance**: Response time tracking
- **Error Logging**: Console logs với timestamp

---

**Lưu ý**: API server phải chạy trước khi activate N8N workflow!