# 🏥 Medical Chatbot - Chatbot Y Tế Thông Minh

Hệ thống chatbot y tế sử dụng Deep Learning để đưa ra gợi ý thuốc không kê đơn (OTC) dựa trên triệu chứng của người dùng.

## 🎯 Tính năng chính

### 🔍 4 Models Deep Learning chính:

1. **Intent Classifier** - Nhận diện ý định người dùng
   - "Tôi bị đau đầu" → `triệu_chứng`
   - "Thuốc Panadol có tác dụng gì?" → `tra_cứu_thuốc`
   - "Tôi có thể uống paracetamol khi bị cảm không?" → `tư_vấn_sử_dụng`

2. **Symptom Extractor (NER)** - Trích xuất triệu chứng và thực thể y tế
   - "Tôi bị đau đầu và sốt nhẹ" → `{đau đầu: SYMPTOM, sốt: SYMPTOM}`

3. **Drug Recommender** - Gợi ý thuốc OTC phù hợp
   - Dựa trên triệu chứng → gợi ý top-K thuốc phù hợp nhất
   - Sử dụng embedding similarity hoặc neural collaborative filtering

4. **Risk Checker** - Kiểm tra rủi ro và chống chỉ định
   - Input: thuốc + profile người dùng (tuổi, giới, bệnh nền)
   - Output: `safe` / `caution` / `contraindicated`

### 🚀 Pipeline hoạt động:
```
User Input → Intent Classification → Entity Extraction → Drug Recommendation → Risk Assessment → Final Response
```

## 📋 Cấu trúc dự án

```
medical-chatbot/
├── src/
│   ├── models/                    # 4 models chính
│   │   ├── intent_classifier.py   # Model 1: Phân loại ý định
│   │   ├── symptom_extractor.py   # Model 2: NER triệu chứng
│   │   ├── drug_recommender.py    # Model 3: Gợi ý thuốc
│   │   └── risk_checker.py        # Model 4: Kiểm tra rủi ro
│   ├── medical_pipeline.py        # Pipeline kết hợp 4 models
│   └── api/
│       └── app.py                 # Flask API server
├── data/
│   ├── processed/                 # Dữ liệu đã xử lý
│   └── models/                    # Models đã train
├── notebooks/                     # Jupyter notebooks
├── frontend/                      # Web interface
├── config.yaml                    # Configuration
├── requirements.txt               # Dependencies
├── demo.py                        # Demo script
└── README.md
```

## 🛠️ Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd medical-chatbot
```

### 2. Tạo virtual environment (khuyến nghị)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Sử dụng

### 1. Demo nhanh
```bash
python demo.py
```

### 2. Chạy API server
```bash
python src/api/app.py
```

### 3. Test với Pipeline
```python
from src.medical_pipeline import MedicalChatbotPipeline

pipeline = MedicalChatbotPipeline()
result = pipeline.process_message("Tôi bị đau đầu")
print(result['final_response'])
```

## ⚠️ Lưu ý quan trọng

- Đây chỉ là công cụ hỗ trợ tham khảo, **KHÔNG THAY THẾ** ý kiến bác sĩ
- Chỉ gợi ý thuốc **không kê đơn (OTC)**
- Luôn khuyên người dùng tham khảo chuyên gia y tế
│   │   └── evaluator.py             # Class for model evaluation
│   ├── api                          # API setup and routes
│   │   ├── app.py                   # Flask application setup
│   │   ├── routes.py                # API routes for chatbot interaction
│   │   └── middleware.py            # Middleware for request handling
│   ├── utils                        # Utility functions
│   │   ├── text_processing.py        # Text processing utilities
│   │   ├── medical_utils.py         # Medical data handling utilities
│   │   └── config.py                # Configuration settings
│   └── chatbot                      # Chatbot conversation management
│       ├── conversation_handler.py   # Manages conversation flow
│       └── response_generator.py     # Generates responses based on input
├── data
│   ├── raw                          # Directory for raw data
│   │   └── .gitkeep
│   ├── processed                    # Directory for processed data
│   │   └── .gitkeep
│   └── models                       # Directory for model files
│       └── .gitkeep
├── notebooks                        # Jupyter notebooks for exploration and training
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── tests                            # Unit tests for the application
│   ├── test_models.py               # Tests for model classes
│   ├── test_api.py                  # Tests for API functionality
│   └── test_utils.py                # Tests for utility functions
├── frontend                         # Frontend interface files
│   ├── index.html                   # Main HTML file
│   ├── style.css                    # Styles for the frontend
│   ├── script.js                    # JavaScript for user interactions
│   └── assets                       # Directory for assets
│       └── .gitkeep
├── requirements.txt                 # Project dependencies
├── setup.py                         # Packaging and dependency management
├── config.yaml                      # Configuration settings in YAML format
├── .gitignore                       # Files to ignore in version control
└── README.md                        # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd medical-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the configuration in `config.yaml` as needed.

## Usage

1. Start the application:
   ```
   python src/main.py
   ```

2. Access the chatbot through the API or the frontend interface.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.