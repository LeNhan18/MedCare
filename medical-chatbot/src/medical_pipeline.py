"""
Medical Chatbot Pipeline
Kết hợp 4 models để tạo thành chatbot y tế hoàn chỉnh
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.intent_classifier import IntentClassifier
from models.symptom_extractor import SymptomExtractor
from models.drug_recommender import DrugRecommender
from models.risk_checker import RiskChecker
import yaml
import json

class MedicalChatbotPipeline:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        
        # Initialize all models
        print("Đang khởi tạo Medical Chatbot Pipeline...")
        
        try:
            self.intent_classifier = IntentClassifier(config_path)
            print("✅ Intent Classifier đã sẵn sàng")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo Intent Classifier: {e}")
            self.intent_classifier = None
        
        try:
            self.symptom_extractor = SymptomExtractor(config_path)
            print("✅ Symptom Extractor đã sẵn sàng")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo Symptom Extractor: {e}")
            self.symptom_extractor = None
        
        try:
            self.drug_recommender = DrugRecommender(config_path)
            print("✅ Drug Recommender đã sẵn sàng")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo Drug Recommender: {e}")
            self.drug_recommender = None
        
        try:
            self.risk_checker = RiskChecker(config_path)
            print("✅ Risk Checker đã sẵn sàng")
        except Exception as e:
            print(f"❌ Lỗi khởi tạo Risk Checker: {e}")
            self.risk_checker = None
        
        print("🚀 Medical Chatbot Pipeline đã được khởi tạo!")
    
    def process_message(self, user_message, user_profile=None):
        """
        Xử lý tin nhắn từ người dùng qua pipeline 4 models
        
        Args:
            user_message (str): Tin nhắn từ người dùng
            user_profile (dict): Thông tin người dùng (tuổi, giới, bệnh nền, thuốc đang dùng)
        
        Returns:
            dict: Kết quả xử lý từ pipeline
        """
        print(f"📨 Xử lý tin nhắn: '{user_message}'")
        
        result = {
            'user_message': user_message,
            'user_profile': user_profile,
            'processing_steps': [],
            'final_response': '',
            'recommendations': [],
            'warnings': []
        }
        
        try:
            # Step 1: Intent Classification
            print("🔍 Step 1: Phân loại ý định...")
            if self.intent_classifier:
                intent_result = self.intent_classifier.predict(user_message)
                result['intent'] = intent_result
                result['processing_steps'].append({
                    'step': 1,
                    'name': 'Intent Classification',
                    'result': intent_result
                })
                print(f"   Intent: {intent_result['intent']} (confidence: {intent_result['confidence']:.3f})")
            else:
                # Fallback: simple rule-based intent detection
                intent_result = self._fallback_intent_detection(user_message)
                result['intent'] = intent_result
                print(f"   Intent (fallback): {intent_result['intent']}")
            
            # Step 2: Entity Extraction (nếu intent liên quan đến triệu chứng)
            entities = []
            if result['intent']['intent'] in ['triệu_chứng', 'tư_vấn_sử_dụng']:
                print("🏷️ Step 2: Trích xuất thực thể y tế...")
                if self.symptom_extractor:
                    entity_result = self.symptom_extractor.predict(user_message)
                    entities = entity_result['entities']
                    result['entities'] = entity_result
                    result['processing_steps'].append({
                        'step': 2,
                        'name': 'Entity Extraction',
                        'result': entity_result
                    })
                    print(f"   Tìm thấy {len(entities)} entities:")
                    for entity in entities:
                        print(f"     - {entity['text']} ({entity['entity']})")
                else:
                    # Fallback: simple keyword extraction
                    entities = self._fallback_entity_extraction(user_message)
                    result['entities'] = {'entities': entities}
                    print(f"   Entities (fallback): {[e['text'] for e in entities]}")
            
            # Step 3: Drug Recommendation (nếu có triệu chứng)
            drug_recommendations = []
            if result['intent']['intent'] in ['triệu_chứng'] and entities:
                print("💊 Step 3: Gợi ý thuốc...")
                if self.drug_recommender:
                    # Tạo text từ các symptoms được trích xuất
                    symptom_texts = []
                    for entity in entities:
                        if entity['entity'] == 'SYMPTOM':
                            symptom_texts.append(entity['text'])
                    
                    if symptom_texts:
                        symptoms_combined = ', '.join(symptom_texts)
                        drug_recommendations = self.drug_recommender.recommend_drugs(
                            symptoms_combined, user_profile
                        )
                        
                        result['drug_recommendations'] = drug_recommendations
                        result['processing_steps'].append({
                            'step': 3,
                            'name': 'Drug Recommendation',
                            'result': drug_recommendations
                        })
                        print(f"   Gợi ý {len(drug_recommendations)} thuốc:")
                        for drug in drug_recommendations:
                            print(f"     - {drug['drug_name']} (score: {drug.get('similarity_score', 'N/A')})")
            
            # Step 4: Risk Checking (cho từng thuốc được gợi ý)
            risk_results = []
            if drug_recommendations and user_profile:
                print("⚠️ Step 4: Kiểm tra rủi ro...")
                if self.risk_checker:
                    for drug in drug_recommendations:
                        risk_result = self.risk_checker.check_risk_rule_based(
                            drug['drug_name'], user_profile
                        )
                        drug['risk_assessment'] = risk_result
                        risk_results.append({
                            'drug_name': drug['drug_name'],
                            'risk_result': risk_result
                        })
                        
                        print(f"     {drug['drug_name']}: {risk_result['risk_level']}")
                        if risk_result['warnings']:
                            for warning in risk_result['warnings']:
                                print(f"       ⚠️ {warning['description']}")
                    
                    result['risk_assessments'] = risk_results
                    result['processing_steps'].append({
                        'step': 4,
                        'name': 'Risk Assessment',
                        'result': risk_results
                    })
            
            # Generate final response
            result['final_response'] = self._generate_response(result)
            print(f"✅ Hoàn thành xử lý!")
            
        except Exception as e:
            print(f"❌ Lỗi trong pipeline: {e}")
            result['error'] = str(e)
            result['final_response'] = "Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu của bạn. Vui lòng thử lại."
        
        return result
    
    def _fallback_intent_detection(self, message):
        """Phát hiện intent đơn giản khi không có model"""
        message_lower = message.lower()
        
        # Keywords cho từng intent
        symptom_keywords = ['bị', 'đau', 'sốt', 'ho', 'nghẹt', 'chảy', 'ngứa', 'buồn nôn', 'tiêu chảy']
        drug_info_keywords = ['thuốc', 'paracetamol', 'ibuprofen', 'tác dụng', 'công dụng']
        usage_keywords = ['có thể', 'uống', 'dùng', 'sử dụng', 'khi nào']
        disease_keywords = ['bệnh', 'nguy hiểm', 'triệu chứng của']
        
        if any(keyword in message_lower for keyword in symptom_keywords):
            return {'intent': 'triệu_chứng', 'confidence': 0.8}
        elif any(keyword in message_lower for keyword in drug_info_keywords):
            return {'intent': 'tra_cứu_thuốc', 'confidence': 0.8}
        elif any(keyword in message_lower for keyword in usage_keywords):
            return {'intent': 'tư_vấn_sử_dụng', 'confidence': 0.8}
        elif any(keyword in message_lower for keyword in disease_keywords):
            return {'intent': 'thông_tin_bệnh', 'confidence': 0.8}
        else:
            return {'intent': 'khác', 'confidence': 0.6}
    
    def _fallback_entity_extraction(self, message):
        """Trích xuất entity đơn giản khi không có model"""
        message_lower = message.lower()
        entities = []
        
        # Common symptoms
        symptom_dict = {
            'đau đầu': 'SYMPTOM',
            'sốt': 'SYMPTOM',
            'ho': 'SYMPTOM',
            'nghẹt mũi': 'SYMPTOM',
            'chảy nước mũi': 'SYMPTOM',
            'đau bụng': 'SYMPTOM',
            'buồn nôn': 'SYMPTOM',
            'tiêu chảy': 'SYMPTOM',
            'ngứa': 'SYMPTOM',
            'viêm': 'SYMPTOM'
        }
        
        for symptom, entity_type in symptom_dict.items():
            if symptom in message_lower:
                entities.append({
                    'text': symptom,
                    'entity': entity_type
                })
        
        return entities
    
    def _generate_response(self, result):
        """Tạo phản hồi cuối cùng dựa trên kết quả pipeline"""
        intent = result['intent']['intent']
        response_parts = []
        
        if intent == 'triệu_chứng':
            # Phản hồi cho triệu chứng
            entities = result.get('entities', {}).get('entities', [])
            symptoms = [e['text'] for e in entities if e['entity'] == 'SYMPTOM']
            
            if symptoms:
                response_parts.append(f"Tôi hiểu bạn đang gặp các triệu chứng: {', '.join(symptoms)}.")
                
                # Thêm gợi ý thuốc nếu có
                drug_recommendations = result.get('drug_recommendations', [])
                if drug_recommendations:
                    response_parts.append("\n🏥 **Gợi ý thuốc OTC có thể giúp:**")
                    
                    for i, drug in enumerate(drug_recommendations, 1):
                        drug_info = f"\n{i}. **{drug['drug_name']}** ({drug['active_ingredient']})"
                        drug_info += f"\n   - Công dụng: {drug['medical_condition']}"
                        drug_info += f"\n   - Liều dùng: {drug['dosage']}"
                        
                        # Thêm đánh giá rủi ro nếu có
                        risk_assessment = drug.get('risk_assessment')
                        if risk_assessment:
                            if risk_assessment['risk_level'] == 'contraindicated':
                                drug_info += f"\n   - ❌ **KHÔNG NÊN DÙNG** - Có chống chỉ định"
                            elif risk_assessment['risk_level'] == 'caution':
                                drug_info += f"\n   - ⚠️ **Cần thận trọng** - Có cảnh báo"
                            else:
                                drug_info += f"\n   - ✅ Có thể sử dụng an toàn"
                        
                        response_parts.append(drug_info)
                    
                    response_parts.append("\n⚠️ **Lưu ý quan trọng:**")
                    response_parts.append("- Đây chỉ là gợi ý thuốc không kê đơn (OTC)")
                    response_parts.append("- Đọc kỹ hướng dẫn sử dụng trước khi dùng")
                    response_parts.append("- Nếu triệu chứng kéo dài hoặc nặng hơn, hãy đến gặp bác sĩ")
                else:
                    response_parts.append("\nTôi chưa thể gợi ý thuốc cụ thể. Bạn nên tham khảo ý kiến dược sĩ hoặc bác sĩ.")
            else:
                response_parts.append("Tôi chưa rõ triệu chứng cụ thể bạn đang gặp. Bạn có thể mô tả chi tiết hơn không?")
        
        elif intent == 'tra_cứu_thuốc':
            response_parts.append("Bạn muốn tra cứu thông tin về thuốc nào? Tôi có thể cung cấp thông tin về:")
            response_parts.append("- Công dụng và cách sử dụng")
            response_parts.append("- Liều lượng khuyến nghị")
            response_parts.append("- Chống chỉ định và cảnh báo")
            response_parts.append("- Tương tác thuốc")
        
        elif intent == 'tư_vấn_sử_dụng':
            response_parts.append("Để tư vấn về việc sử dụng thuốc, tôi cần biết:")
            response_parts.append("- Thuốc bạn muốn sử dụng")
            response_parts.append("- Triệu chứng hiện tại")
            response_parts.append("- Độ tuổi và thông tin sức khỏe")
            response_parts.append("- Các thuốc khác đang sử dụng (nếu có)")
        
        elif intent == 'thông_tin_bệnh':
            response_parts.append("Tôi có thể cung cấp thông tin cơ bản về các bệnh phổ biến.")
            response_parts.append("Tuy nhiên, để chẩn đoán chính xác và điều trị hiệu quả, ")
            response_parts.append("bạn nên tham khảo ý kiến của bác sĩ chuyên khoa.")
        
        else:
            response_parts.append("Xin chào! Tôi là chatbot y tế, có thể giúp bạn:")
            response_parts.append("- Gợi ý thuốc OTC cho các triệu chứng thường gặp")
            response_parts.append("- Tra cứu thông tin thuốc")
            response_parts.append("- Tư vấn sử dụng thuốc an toàn")
            response_parts.append("- Cung cấp thông tin bệnh cơ bản")
            response_parts.append("\nBạn có triệu chứng gì cần tư vấn không?")
        
        return ''.join(response_parts)
    
    def load_all_models(self):
        """Load tất cả models đã được huấn luyện"""
        print("📂 Đang load các models...")
        
        if self.intent_classifier:
            try:
                self.intent_classifier.load_model()
                print("✅ Intent Classifier loaded")
            except Exception as e:
                print(f"❌ Không thể load Intent Classifier: {e}")
        
        if self.symptom_extractor:
            try:
                self.symptom_extractor.load_model()
                print("✅ Symptom Extractor loaded")
            except Exception as e:
                print(f"❌ Không thể load Symptom Extractor: {e}")
        
        if self.drug_recommender:
            try:
                self.drug_recommender.load_model()
                print("✅ Drug Recommender loaded")
            except Exception as e:
                print(f"❌ Không thể load Drug Recommender: {e}")
        
        if self.risk_checker:
            try:
                self.risk_checker.load_model()
                print("✅ Risk Checker loaded")
            except Exception as e:
                print(f"❌ Không thể load Risk Checker: {e}")
    
    def get_pipeline_status(self):
        """Lấy trạng thái của pipeline"""
        status = {
            'intent_classifier': self.intent_classifier is not None,
            'symptom_extractor': self.symptom_extractor is not None,
            'drug_recommender': self.drug_recommender is not None,
            'risk_checker': self.risk_checker is not None
        }
        
        active_models = sum(status.values())
        status['total_active'] = active_models
        status['ready'] = active_models >= 2  # Cần ít nhất 2 models để hoạt động cơ bản
        
        return status


# Example usage và test
if __name__ == "__main__":
    # Test cases
    test_messages = [
        "Tôi bị đau đầu và sốt",
        "Thuốc Paracetamol có tác dụng gì?",
        "Tôi có thể uống ibuprofen khi bị cảm không?",
        "Bệnh tiểu đường có nguy hiểm không?",
        "Xin chào"
    ]
    
    # Test user profile
    test_profile = {
        'age': 28,
        'gender': 'female',
        'existing_conditions': ['dị ứng aspirin'],
        'current_drugs': [],
        'pregnancy': {'is_pregnant': False}
    }
    
    # Initialize pipeline
    print("🤖 Khởi tạo Medical Chatbot Pipeline...")
    pipeline = MedicalChatbotPipeline()
    
    # Check status
    status = pipeline.get_pipeline_status()
    print(f"\n📊 Trạng thái Pipeline:")
    print(f"   - Intent Classifier: {'✅' if status['intent_classifier'] else '❌'}")
    print(f"   - Symptom Extractor: {'✅' if status['symptom_extractor'] else '❌'}")
    print(f"   - Drug Recommender: {'✅' if status['drug_recommender'] else '❌'}")
    print(f"   - Risk Checker: {'✅' if status['risk_checker'] else '❌'}")
    print(f"   - Pipeline Ready: {'✅' if status['ready'] else '❌'}")
    
    # Test với một message
    if status['ready']:
        print(f"\n🧪 Test với message đầu tiên...")
        test_message = test_messages[0]
        result = pipeline.process_message(test_message, test_profile)
        
        print(f"\n📋 Kết quả:")
        print(f"Intent: {result['intent']['intent']}")
        print(f"Entities: {len(result.get('entities', {}).get('entities', []))}")
        print(f"Drug recommendations: {len(result.get('drug_recommendations', []))}")
        print(f"Risk assessments: {len(result.get('risk_assessments', []))}")
        
        print(f"\n💬 Response:")
        print(result['final_response'])
    else:
        print("❌ Pipeline chưa sẵn sàng để test")