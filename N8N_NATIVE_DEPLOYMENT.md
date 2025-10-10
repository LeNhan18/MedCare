# 🚀 Hướng dẫn triển khai Medical Chatbot trực tiếp trên N8N

## 📝 Tổng quan

Workflow này chạy **hoàn toàn trên N8N** mà không cần API server riêng:
- ✅ Xử lý tin nhắn tiếng Việt
- ✅ Phát hiện triệu chứng thông minh  
- ✅ Tìm kiếm thuốc phù hợp
- ✅ Tạo phản hồi tự nhiên
- ✅ Không cần cài đặt gì thêm

## 🛠️ Cách triển khai

### Bước 1: Import Workflow
```bash
1. Mở N8N interface
2. Click "Import from file" 
3. Chọn: medical-chatbot-workflow-native.json
4. Click "Import"
```

### Bước 2: Activate Workflow  
```bash
1. Click nút "Active" để bật workflow
2. Copy "Webhook URL" từ node đầu tiên
3. URL sẽ có dạng: https://your-n8n.com/webhook/medical-chat
```

### Bước 3: Test ngay lập tức
```powershell
# Test với PowerShell
$body = @{
    message = "Tôi bị đau đầu và sốt"
    userId = "test_user"
    sessionId = "test_session"
} | ConvertTo-Json

Invoke-RestMethod -Uri "YOUR_WEBHOOK_URL" -Method POST -Body $body -ContentType "application/json"
```

## 🧠 Workflow Logic

### 1. **Webhook Input**
```json
{
  "message": "Tôi bị đau đầu rất nhiều",
  "userId": "user123", 
  "sessionId": "session456"
}
```

### 2. **Symptom Detection** 
- Phát hiện: `["đau_đầu"]`
- Severity: `"severe"` (vì có từ "rất nhiều")
- Hỗ trợ 10+ triệu chứng phổ biến

### 3. **Drug Search**
- Tìm thuốc phù hợp từ database embed
- Ưu tiên OTC cho triệu chứng nhẹ
- Sắp xếp theo rating

### 4. **Response Generation**
```
Dựa trên triệu chứng "đau đầu" (mức độ: nặng), tôi gợi ý:

**1. Paracetamol** 🟢 OTC
   📋 Dùng cho: đau đầu
   💊 Loại thuốc: thuốc giảm đau
   ⭐⭐⭐⭐ Đánh giá: 8.5/10
   ⚠️ Tác dụng phụ: buồn nôn, chóng mặt

🛡️ **LƯU Ý QUAN TRỌNG:**
• 🟢 OTC: Không cần toa bác sĩ
• Thông tin này chỉ mang tính chất tham khảo
• Tham khảo ý kiến bác sĩ/dược sĩ trước khi sử dụng

🚨 **CẢNH BÁO**: Triệu chứng nặng - Nên thăm khám bác sĩ ngay!
```

## 🎯 Tính năng thông minh

### **Symptom Detection**
```javascript
// Hỗ trợ các triệu chứng:
đau_đầu: ['đau đầu', 'nhức đầu', 'migraine']
sốt: ['sốt', 'fever', 'nóng người', 'ốm']  
ho: ['ho', 'cough', 'ho khan', 'ho có đờm']
đau_bụng: ['đau bụng', 'stomach pain', 'đau dạ dày']
// ... và nhiều hơn nữa
```

### **Severity Analysis**
- **Severe**: "rất", "nhiều", "nặng", "khủng khiếp"
- **Mild**: "hơi", "nhẹ", "ít"  
- **Moderate**: mặc định

### **Drug Prioritization**
- OTC drugs cho triệu chứng nhẹ
- Prescription drugs cho triệu chứng nặng
- Sắp xếp theo rating cao nhất

## 🔧 Customization

### Thêm triệu chứng mới:
```javascript
// Trong Symptom Detection node
const symptomKeywords = {
  // Thêm triệu chứng mới
  'đau_khớp': ['đau khớp', 'arthritis', 'joint pain'],
  'trầm_cảm': ['trầm cảm', 'depression', 'buồn chán']
};
```

### Thêm thuốc mới:
```javascript
// Trong Drug Search Engine node  
const medicalDataSample = [
  // Thêm thuốc mới
  {
    "drug_name": "Thuốc Mới",
    "medical_condition_vi": "triệu chứng mới", 
    "side_effects_vi": "tác dụng phụ",
    "rx_otc": "OTC",
    "rating": 8.0
  }
];
```

## 📱 Integration Examples

### **Website Integration**
```html
<script>
async function sendMessage(message) {
  const response = await fetch('YOUR_WEBHOOK_URL', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: message,
      userId: 'web_user_' + Date.now(),
      sessionId: 'web_session_' + Date.now()
    })
  });
  
  const result = await response.json();
  document.getElementById('response').innerHTML = result.response;
}
</script>
```

### **Mobile App (Flutter)**
```dart
Future<void> sendMessage(String message) async {
  final response = await http.post(
    Uri.parse('YOUR_WEBHOOK_URL'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'message': message,
      'userId': 'mobile_user_${DateTime.now().millisecondsSinceEpoch}',
      'sessionId': 'mobile_session_${DateTime.now().millisecondsSinceEpoch}',
    }),
  );
  
  final result = jsonDecode(response.body);
  print(result['response']);
}
```

### **WhatsApp/Telegram Bot**
```javascript
// Kết nối với WhatsApp/Telegram API
const chatbotResponse = await fetch('YOUR_WEBHOOK_URL', {
  method: 'POST', 
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: userMessage,
    userId: `whatsapp_${phoneNumber}`,
    sessionId: `whatsapp_session_${Date.now()}`
  })
});
```

## 🧪 Test Cases

### Test 1: Triệu chứng rõ ràng
```json
{
  "message": "Tôi bị đau đầu rất nặng",
  "userId": "test1",
  "sessionId": "session1"
}
```
**Expected**: Trả về thuốc đau đầu với severity="severe"

### Test 2: Nhiều triệu chứng
```json
{
  "message": "Tôi bị sốt, ho và chóng mặt",
  "userId": "test2", 
  "sessionId": "session2"
}
```
**Expected**: Trả về thuốc cho cả 3 triệu chứng

### Test 3: Không có triệu chứng rõ ràng
```json
{
  "message": "Tôi cảm thấy không khỏe",
  "userId": "test3",
  "sessionId": "session3" 
}
```
**Expected**: Hỏi lại để làm rõ triệu chứng

## 🔄 Workflow Monitoring

N8N cung cấp dashboard monitoring:
- ✅ Execution history
- ✅ Error tracking
- ✅ Performance metrics
- ✅ Real-time logs

## 🚀 Production Tips

### Performance:
- Workflow chạy serverless trên N8N cloud
- Response time ~200-500ms
- Auto-scaling theo traffic

### Security:
- Webhook URL là random/private  
- Có thể thêm API key authentication
- Request logging tự động

### Scalability:
- N8N cloud handle unlimited requests
- Có thể duplicate workflow cho A/B testing
- Easy backup/restore workflow

## ✅ Advantages của N8N Native

1. **Zero Setup**: Không cần server riêng
2. **Visual Flow**: Dễ debug và modify
3. **Auto Scaling**: N8N cloud tự động scale
4. **Rich Integrations**: Kết nối với 200+ services
5. **Real-time Monitoring**: Dashboard built-in

---

**🎉 Workflow đã sẵn sàng! Import và test ngay!**

**Webhook URL Example**: 
`https://your-n8n-instance.com/webhook/medical-chat`