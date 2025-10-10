"""
Demo script để test Medical Chatbot Pipeline
Chạy script này để xem chatbot hoạt động như thế nào
"""

import sys
import os

# Thêm đường dẫn để import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_banner():
    """In banner chào mừng"""
    print("=" * 80)
    print("🏥 MEDICAL CHATBOT DEMO")
    print("Chatbot Y Tế Thông Minh với Deep Learning")
    print("=" * 80)
    print()

def print_step(step_num, title):
    """In tiêu đề từng bước"""
    print(f"\n{'='*10} BƯỚC {step_num}: {title.upper()} {'='*10}")

def demo_basic_functionality():
    """Demo chức năng cơ bản không cần models"""
    print_step(1, "Demo chức năng cơ bản")
    
    # Test messages
    test_cases = [
        {
            'message': 'Tôi bị đau đầu và sốt',
            'user_profile': {
                'age': 25,
                'gender': 'female',
                'existing_conditions': [],
                'current_drugs': []
            },
            'description': 'Triệu chứng cơ bản'
        },
        {
            'message': 'Thuốc Paracetamol có tác dụng gì?',
            'user_profile': None,
            'description': 'Hỏi thông tin thuốc'
        },
        {
            'message': 'Tôi có thể uống ibuprofen khi bị cảm không?',
            'user_profile': {
                'age': 30,
                'gender': 'male',
                'existing_conditions': ['hen suyễn'],
                'current_drugs': ['salbutamol']
            },
            'description': 'Tư vấn sử dụng thuốc với bệnh nền'
        }
    ]
    
    print("📋 Test cases được chuẩn bị:")
    for i, case in enumerate(test_cases, 1):
        print(f"  {i}. {case['description']}: '{case['message']}'")
    
    return test_cases

def demo_drug_database():
    """Demo database thuốc"""
    print_step(2, "Demo cơ sở dữ liệu thuốc")
    
    try:
        from models.drug_recommender import DrugRecommender
        
        recommender = DrugRecommender()
        print(f"✅ Đã load database với {len(recommender.drug_database)} thuốc")
        
        print("\n📋 Một số thuốc trong database:")
        sample_drugs = recommender.drug_database.head(5)
        for _, drug in sample_drugs.iterrows():
            print(f"  - {drug['drug_name']} ({drug['active_ingredient']})")
            print(f"    Điều trị: {drug['symptoms']}")
            print(f"    Chống chỉ định: {drug['contraindications']}")
            print()
            
        return True
        
    except Exception as e:
        print(f"❌ Không thể load database: {e}")
        return False

def demo_risk_checking():
    """Demo risk checking"""
    print_step(3, "Demo kiểm tra rủi ro")
    
    try:
        from models.risk_checker import RiskChecker
        
        checker = RiskChecker()
        print(f"✅ Đã load risk checker với {len(checker.risk_database)} quy tắc")
        
        # Test risk checking
        test_profile = {
            'age': 5,
            'gender': 'male',
            'existing_conditions': [],
            'current_drugs': []
        }
        
        test_drugs = ['Paracetamol', 'Ibuprofen', 'Aspirin']
        
        print(f"\n🧪 Test với profile: Trẻ nam 5 tuổi")
        for drug in test_drugs:
            risk_result = checker.check_risk_rule_based(drug, test_profile)
            print(f"\n  {drug}:")
            print(f"    - Mức độ rủi ro: {risk_result['risk_level']}")
            print(f"    - Confidence: {risk_result['confidence']:.2f}")
            if risk_result['warnings']:
                print(f"    - Cảnh báo: {len(risk_result['warnings'])} vấn đề")
                for warning in risk_result['warnings'][:2]:  # Show first 2 warnings
                    print(f"      • {warning['description']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Không thể test risk checking: {e}")
        return False

def demo_pipeline():
    """Demo pipeline hoàn chỉnh"""
    print_step(4, "Demo Pipeline hoàn chỉnh")
    
    try:
        from medical_pipeline import MedicalChatbotPipeline
        
        pipeline = MedicalChatbotPipeline()
        status = pipeline.get_pipeline_status()
        
        print("📊 Trạng thái Pipeline:")
        for component, active in status.items():
            if component != 'total_active' and component != 'ready':
                status_icon = "✅" if active else "❌"
                print(f"  {status_icon} {component.replace('_', ' ').title()}")
        
        print(f"\n🚀 Pipeline ready: {'✅' if status['ready'] else '❌'}")
        
        if status['ready']:
            # Test với một message
            test_message = "Tôi bị đau đầu và sốt"
            test_profile = {
                'age': 28,
                'gender': 'female',
                'existing_conditions': ['dị ứng aspirin'],
                'current_drugs': []
            }
            
            print(f"\n🧪 Test message: '{test_message}'")
            print(f"👤 User profile: {test_profile['age']} tuổi, {test_profile['gender']}, dị ứng aspirin")
            
            result = pipeline.process_message(test_message, test_profile)
            
            print(f"\n📋 Kết quả xử lý:")
            print(f"  - Intent: {result['intent']['intent']} (confidence: {result['intent']['confidence']:.2f})")
            
            entities = result.get('entities', {}).get('entities', [])
            print(f"  - Entities: {len(entities)} thực thể")
            for entity in entities:
                print(f"    • {entity['text']} ({entity['entity']})")
            
            drugs = result.get('drug_recommendations', [])
            print(f"  - Drug recommendations: {len(drugs)} thuốc")
            for drug in drugs[:3]:  # Show first 3
                print(f"    • {drug['drug_name']} - {drug['medical_condition']}")
            
            risks = result.get('risk_assessments', [])
            print(f"  - Risk assessments: {len(risks)} đánh giá")
            
            print(f"\n💬 Response preview:")
            response_preview = result['final_response'][:200] + "..." if len(result['final_response']) > 200 else result['final_response']
            print(f"  {response_preview}")
            
            return True
        else:
            print("⚠️ Pipeline chưa sẵn sàng để demo đầy đủ")
            return False
            
    except Exception as e:
        print(f"❌ Không thể demo pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_api_simulation():
    """Demo simulation API calls"""
    print_step(5, "Demo API Simulation")
    
    print("🌐 Simulation API calls (không cần Flask server):")
    
    # Simulate API requests
    api_requests = [
        {
            'endpoint': 'POST /chat',
            'data': {
                'message': 'Tôi bị đau đầu',
                'user_profile': {'age': 25, 'gender': 'female'}
            }
        },
        {
            'endpoint': 'POST /drug-info',
            'data': {
                'drug_name': 'Paracetamol'
            }
        },
        {
            'endpoint': 'POST /risk-check',
            'data': {
                'drug_name': 'Ibuprofen',
                'user_profile': {
                    'age': 5,
                    'existing_conditions': ['hen suyễn']
                }
            }
        }
    ]
    
    for i, req in enumerate(api_requests, 1):
        print(f"\n  {i}. {req['endpoint']}")
        print(f"     Request: {req['data']}")
        print(f"     Response: Sẽ trả về JSON với thông tin tương ứng")

def main():
    """Main demo function"""
    print_banner()
    
    print("🎯 Demo này sẽ cho bạn thấy:")
    print("  1. Cơ sở dữ liệu thuốc và rủi ro")
    print("  2. Các components riêng lẻ")
    print("  3. Pipeline hoàn chỉnh")
    print("  4. API simulation")
    print()
    input("Nhấn Enter để bắt đầu...")
    
    # Run demos
    results = []
    
    # Demo 1: Basic functionality
    test_cases = demo_basic_functionality()
    results.append(True)
    
    # Demo 2: Drug database
    db_result = demo_drug_database()
    results.append(db_result)
    
    # Demo 3: Risk checking
    risk_result = demo_risk_checking()
    results.append(risk_result)
    
    # Demo 4: Full pipeline
    pipeline_result = demo_pipeline()
    results.append(pipeline_result)
    
    # Demo 5: API simulation
    demo_api_simulation()
    results.append(True)
    
    # Summary
    print("\n" + "="*50)
    print("📊 KẾT QUẢ DEMO")
    print("="*50)
    
    demo_names = [
        "Chức năng cơ bản",
        "Database thuốc", 
        "Kiểm tra rủi ro",
        "Pipeline hoàn chỉnh",
        "API simulation"
    ]
    
    for i, (name, result) in enumerate(zip(demo_names, results), 1):
        status = "✅ Thành công" if result else "❌ Thất bại"
        print(f"  {i}. {name}: {status}")
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n🎯 Tổng kết: {success_count}/{total_count} demos thành công")
    
    if success_count >= 3:
        print("\n🎉 Hệ thống Medical Chatbot đã sẵn sàng!")
        print("\n🚀 Các bước tiếp theo:")
        print("  1. Cài đặt dependencies: pip install -r requirements.txt")
        print("  2. Chạy API server: python src/api/app.py")
        print("  3. Test API qua browser hoặc Postman")
        print("  4. Huấn luyện models với dữ liệu thực tế")
    else:
        print("\n⚠️ Một số components chưa hoạt động đầy đủ.")
        print("Kiểm tra lại dependencies và configuration.")
    
    print("\n💡 Ghi chú:")
    print("  - Models hiện tại dùng rule-based và sample data")
    print("  - Để có hiệu quả tốt nhất, cần train với dữ liệu y tế thực tế")
    print("  - Luôn có disclaimer về tính chất tham khảo của thông tin")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()