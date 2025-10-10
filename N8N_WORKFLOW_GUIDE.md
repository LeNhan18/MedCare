# 🤖 Hướng dẫn tạo Medical Chatbot trên N8N

## 📋 Tổng quan Workflow

Workflow này sẽ xử lý trực tiếp trên N8N mà không cần API server riêng:

```
Webhook → Phân tích tin nhắn → Tìm kiếm thuốc → Tạo phản hồi → Gửi kết quả
```

## 🚀 Các bước thực hiện

### Bước 1: Import Workflow
1. Mở N8N interface
2. Click **"Import from file"**
3. Chọn file: `n8n-workflows/medical-chatbot-workflow-v2.json`
4. Click **"Import"**

### Bước 2: Cấu hình Webhook
1. Click vào node **"Webhook"**
2. Thiết lập:
   - **HTTP Method**: POST
   - **Path**: `/medical-chat`
   - **Response Mode**: Respond to Webhook
3. Copy **Webhook URL** để test

### Bước 3: Upload dữ liệu thuốc
1. Vào node **"Load Medical Data"**
2. Trong **"Execute Workflow"** tab:
   - Upload file `data/medical_dataset_training.json`
   - Hoặc copy nội dung JSON vào **"JSON"** tab

### Bước 4: Test Workflow
Gửi POST request đến Webhook URL:

```json
{
  "message": "Tôi bị đau đầu và sốt",
  "userId": "test_user",
  "sessionId": "test_session"
}
```

## 🔧 Cấu hình chi tiết từng Node

### 1. **Webhook Node**
```javascript
// Input Schema
{
  "message": "string",
  "userId": "string", 
  "sessionId": "string"
}
```

### 2. **Symptom Detection Node** (Code Node)
```javascript
// Phát hiện triệu chứng từ tin nhắn
const message = $input.first().json.message.toLowerCase();

const symptomKeywords = {
  'đau_đầu': ['đau đầu', 'nhức đầu', 'migraine', 'headache'],
  'sốt': ['sốt', 'fever', 'nóng người', 'ốm'],
  'ho': ['ho', 'cough', 'ho khan', 'ho có đờm'],
  'đau_bụng': ['đau bụng', 'stomach pain', 'đau dạ dày'],
  'cảm_lạnh': ['cảm lạnh', 'cold', 'flu', 'cúm'],
  'mụn_trứng_cá': ['mụn', 'acne', 'mụn trứng cá', 'da mụn'],
  'tiêu_chảy': ['tiêu chảy', 'diarrhea', 'đi lỏng'],
  'táo_bón': ['táo bón', 'constipation', 'khó đi đại tiện'],
  'viêm_họng': ['viêm họng', 'sore throat', 'đau họng'],
  'chóng_mặt': ['chóng mặt', 'dizzy', 'hoa mắt']
};

let detectedSymptoms = [];
let severity = 'mild';

// Phát hiện triệu chứng
for (let symptom in symptomKeywords) {
  for (let keyword of symptomKeywords[symptom]) {
    if (message.includes(keyword)) {
      detectedSymptoms.push(symptom);
      break;
    }
  }
}

// Xác định độ nặng
if (message.includes('rất') || message.includes('nhiều') || message.includes('nặng')) {
  severity = 'severe';
} else if (message.includes('hơi') || message.includes('nhẹ')) {
  severity = 'mild';
} else {
  severity = 'moderate';
}

return {
  json: {
    originalMessage: $input.first().json.message,
    detectedSymptoms: detectedSymptoms,
    severity: severity,
    userId: $input.first().json.userId,
    sessionId: $input.first().json.sessionId
  }
};
```

### 3. **Drug Search Node** (Code Node)
```javascript
// Tìm kiếm thuốc từ dataset
const symptoms = $input.first().json.detectedSymptoms;
const severity = $input.first().json.severity;

// Load medical dataset (cần upload trước)
const medicalData = [
  // Dữ liệu từ medical_dataset_training.json sẽ được load ở đây
  // Hoặc connect với HTTP Request node để fetch từ file
];

let foundDrugs = [];

// Tìm kiếm thuốc phù hợp
for (let symptom of symptoms) {
  const matchingDrugs = medicalData.filter(drug => {
    const condition = drug.medical_condition_vi?.toLowerCase() || '';
    const conditionEn = drug.medical_condition?.toLowerCase() || '';
    
    return condition.includes(symptom.replace('_', ' ')) || 
           conditionEn.includes(symptom.replace('_', ' '));
  });
  
  // Sắp xếp theo rating và ưu tiên OTC cho severity nhẹ
  matchingDrugs.sort((a, b) => {
    if (severity === 'mild') {
      if (a.rx_otc === 'OTC' && b.rx_otc !== 'OTC') return -1;
      if (b.rx_otc === 'OTC' && a.rx_otc !== 'OTC') return 1;
    }
    return (b.rating || 0) - (a.rating || 0);
  });
  
  foundDrugs.push(...matchingDrugs.slice(0, 3));
}

// Loại bỏ duplicate
const uniqueDrugs = foundDrugs.filter((drug, index, self) => 
  index === self.findIndex(d => d.drug_name === drug.drug_name)
);

return {
  json: {
    symptoms: symptoms,
    severity: severity,
    recommendedDrugs: uniqueDrugs.slice(0, 5),
    totalFound: uniqueDrugs.length,
    userId: $input.first().json.userId,
    sessionId: $input.first().json.sessionId
  }
};
```

### 4. **Response Generator Node** (Code Node)
```javascript
// Tạo phản hồi thân thiện
const symptoms = $input.first().json.symptoms;
const drugs = $input.first().json.recommendedDrugs;
const severity = $input.first().json.severity;

let response = "";

if (symptoms.length === 0) {
  response = "Tôi không thể xác định triệu chứng cụ thể từ tin nhắn của bạn. Bạn có thể mô tả rõ hơn về triệu chứng không?";
} else if (drugs.length === 0) {
  response = `Tôi đã hiểu bạn có triệu chứng: ${symptoms.join(', ').replace(/_/g, ' ')}. Tuy nhiên, tôi không tìm thấy thuốc phù hợp trong cơ sở dữ liệu. Bạn nên tham khảo ý kiến bác sĩ.`;
} else {
  response = `Dựa trên triệu chứng "${symptoms.join(', ').replace(/_/g, ' ')}" của bạn, tôi gợi ý các thuốc sau:\n\n`;
  
  drugs.forEach((drug, index) => {
    response += `${index + 1}. **${drug.drug_name}**\n`;
    response += `   - Dùng cho: ${drug.condition_vi || drug.medical_condition}\n`;
    response += `   - Loại: ${drug.rx_otc} (${drug.rx_otc === 'OTC' ? 'Không cần toa' : 'Cần toa bác sĩ'})\n`;
    response += `   - Đánh giá: ${drug.rating || 'N/A'}/10\n`;
    if (drug.side_effects_vi) {
      response += `   - Tác dụng phụ: ${drug.side_effects_vi}\n`;
    }
    response += `\n`;
  });
  
  response += `⚠️ **Lưu ý quan trọng:**\n`;
  response += `- Thông tin này chỉ mang tính chất tham khảo\n`;
  response += `- Vui lòng tham khảo ý kiến bác sĩ hoặc dược sĩ trước khi sử dụng\n`;
  response += `- Đọc kỹ hướng dẫn sử dụng trước khi dùng thuốc`;
}

return {
  json: {
    response: response,
    metadata: {
      symptoms: symptoms,
      drugsFound: drugs.length,
      severity: severity,
      timestamp: new Date().toISOString()
    },
    userId: $input.first().json.userId,
    sessionId: $input.first().json.sessionId
  }
};
```

### 5. **Webhook Response Node**
```javascript
// Trả về response cho user
return {
  json: {
    success: true,
    message: $input.first().json.response,
    metadata: $input.first().json.metadata,
    userId: $input.first().json.userId,
    sessionId: $input.first().json.sessionId
  }
};
```

## 📊 Upload Medical Dataset

Có 2 cách upload dữ liệu thuốc:

### Cách 1: Embed trong workflow
1. Copy nội dung `medical_dataset_training.json`
2. Paste vào **Set Node** trong workflow
3. Reference từ các Code nodes

### Cách 2: HTTP Request từ file
1. Host file JSON trên web server local
2. Dùng **HTTP Request Node** fetch data
3. Cache trong workflow memory

## 🧪 Testing Workflow

### Test Cases:
```json
// Test 1: Đau đầu
{
  "message": "Tôi bị đau đầu rất nhiều",
  "userId": "test1",
  "sessionId": "session1"
}

// Test 2: Nhiều triệu chứng
{
  "message": "Tôi bị sốt và ho, cảm thấy rất mệt",
  "userId": "test2", 
  "sessionId": "session2"
}

// Test 3: Triệu chứng không rõ
{
  "message": "Tôi cảm thấy không khỏe",
  "userId": "test3",
  "sessionId": "session3"
}
```

## 🔗 Integration

### Với website:
```javascript
// Frontend call
fetch('YOUR_N8N_WEBHOOK_URL', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: userMessage,
    userId: getCurrentUserId(),
    sessionId: getSessionId()
  })
});
```

### Với mobile app:
```dart
// Flutter/Dart example
final response = await http.post(
  Uri.parse('YOUR_N8N_WEBHOOK_URL'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'message': userMessage,
    'userId': userId,
    'sessionId': sessionId,
  }),
);
```

## 🎯 Tối ưu hóa

### Performance:
- Cache medical dataset trong workflow memory
- Implement fuzzy matching cho triệu chứng
- Add rate limiting cho webhook

### Accuracy:
- Expand symptom keywords dictionary
- Add machine learning scoring
- Implement conversation context

### Security:
- Add API key authentication
- Validate input data
- Implement request logging

## 🚨 Troubleshooting

### Workflow không chạy:
- Kiểm tra tất cả nodes đã connected
- Verify JSON format trong data nodes
- Check webhook URL accessible

### Không tìm được thuốc:
- Verify medical dataset loaded correctly
- Check symptom keywords matching
- Debug symptom detection logic

### Response không đúng format:
- Check response generator code
- Verify webhook response structure
- Test với simple static response

---

**Workflow hoàn chỉnh đã sẵn sàng! Import và test ngay! 🚀**