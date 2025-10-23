#!/usr/bin/env python3
"""
Cân bằng hoàn toàn dataset - đưa tất cả intent lên mức tối thiểu
"""

import pandas as pd
import random
from typing import List, Dict
import json

class DatasetBalancer:
    def __init__(self):
        # Minimum samples cho mỗi intent
        self.min_samples = 150
        
        # Advanced templates cho các intent thiếu
        self.templates = {
            'general_health': [
                "Làm thế nào để {action}?",
                "Cách {action} hiệu quả nhất?",
                "Tôi nên {action} như thế nào?",
                "Bí quyết để {action}?",
                "Phương pháp {action} tốt nhất?",
                "Làm sao để {action} an toàn?",
                "Cần làm gì để {action}?",
                "Quy tắc {action} cơ bản?",
                "Hướng dẫn {action} chi tiết?",
                "Kinh nghiệm {action} hiệu quả?"
            ],
            'greeting': [
                "{greeting} {title}!",
                "{greeting}, tôi cần tư vấn y tế",
                "{greeting}, bạn có thể giúp tôi không?",
                "{greeting} {title}, tôi có câu hỏi",
                "{greeting}, xin chào!",
                "{greeting} {title} ơi",
                "{greeting}, chúc {title} {time}",
                "{greeting}, rất vui được gặp {title}",
                "{greeting} {title}, tôi cần hỗ trợ",
                "{greeting}, cảm ơn {title} đã hỗ trợ"
            ],
            'drug_interaction': [
                "Thuốc {drug1} có tương tác với {drug2} không?",
                "Tôi đang uống {drug1}, có thể uống thêm {drug2}?",
                "Liệu {drug1} và {drug2} có xung đột?",
                "Kết hợp {drug1} với {drug2} có an toàn?",
                "Thuốc {drug1} có ảnh hưởng đến {drug2}?",
                "Đang dùng {drug1}, có được uống {drug2}?",
                "Tương tác giữa {drug1} và {drug2} như thế nào?",
                "Có thể dùng {drug1} cùng với {drug2} không?",
                "Thuốc {drug1} có làm giảm tác dụng của {drug2}?",
                "Phối hợp {drug1} và {drug2} có nguy hiểm không?"
            ],
            'diet_lifestyle': [
                "Chế độ ăn cho người {condition}?",
                "Thực phẩm nên tránh khi {condition}?",
                "Tập thể dục như thế nào khi {condition}?",
                "Sinh hoạt hàng ngày cho {condition}?",
                "Lối sống lành mạnh với {condition}?",
                "Chế độ dinh dưỡng {condition}?",
                "Thói quen tốt cho {condition}?",
                "Cách sống khỏe với {condition}?",
                "Điều chỉnh lối sống khi {condition}?",
                "Bí quyết sống khỏe mạnh với {condition}?"
            ],
            'medical_procedure': [
                "Quy trình {procedure} như thế nào?",
                "Chuẩn bị gì trước khi {procedure}?",
                "Sau {procedure} cần lưu ý gì?",
                "Chi phí {procedure} bao nhiêu?",
                "Thời gian {procedure} mất bao lâu?",
                "Rủi ro của {procedure} là gì?",
                "Hiệu quả của {procedure} ra sao?",
                "Ai nên thực hiện {procedure}?",
                "Khi nào cần {procedure}?",
                "Thay thế cho {procedure} có gì?"
            ]
        }
        
        # Data để fill templates
        self.template_data = {
            'action': [
                'giữ sức khỏe', 'tăng cường miễn dịch', 'giảm stress', 'ngủ ngon',
                'ăn uống lành mạnh', 'tập thể dục', 'phòng bệnh', 'chăm sóc da',
                'bảo vệ mắt', 'giữ dáng', 'thải độc cơ thể', 'tăng cường trí nhớ',
                'cải thiện tuần hoàn', 'giữ ấm mùa đông', 'chống lão hóa'
            ],
            'greeting': [
                'Xin chào', 'Chào', 'Hello', 'Hi', 'Chào bác sĩ', 'Kính chào',
                'Chúc', 'Xin kính chào', 'Chào buổi sáng', 'Chào buổi chiều'
            ],
            'title': [
                'bác sĩ', 'doctor', 'thầy thuốc', 'chuyên gia', 'anh/chị',
                'thầy', 'cô', 'bạn', 'ông/bà', 'quý vị'
            ],
            'time': [
                'buổi sáng tốt lành', 'buổi chiều vui vẻ', 'ngày mới tốt lành',
                'một ngày tuyệt vời', 'sức khỏe', 'may mắn', 'bình an'
            ],
            'drug1': [
                'paracetamol', 'ibuprofen', 'aspirin', 'amoxicillin', 'omeprazole',
                'metformin', 'atorvastatin', 'lisinopril', 'amlodipine', 'losartan'
            ],
            'drug2': [
                'vitamin C', 'calcium', 'iron', 'warfarin', 'digoxin',
                'insulin', 'prednisolone', 'gabapentin', 'tramadol', 'morphine'
            ],
            'condition': [
                'tiểu đường', 'cao huyết áp', 'tim mạch', 'gan nhiễm mỡ',
                'gout', 'cholesterol cao', 'đau khớp', 'hen suyễn', 'dạ dày',
                'thận yếu', 'mất ngủ', 'trầm cảm', 'lo âu', 'béo phì'
            ],
            'procedure': [
                'nội soi dạ dày', 'siêu âm tim', 'chụp CT', 'MRI não',
                'xét nghiệm máu', 'điện tim', 'sinh thiết', 'phẫu thuật',
                'nội soi phế quản', 'đo mật độ xương', 'chụp X-quang',
                'xét nghiệm nước tiểu', 'đo huyết áp', 'test dị ứng'
            ]
        }

    def load_current_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset hiện tại"""
        df = pd.read_csv(file_path)
        print(f"📂 Loaded {len(df)} samples")
        return df

    def analyze_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Phân tích phân phối intent"""
        distribution = df['intent'].value_counts().to_dict()
        print("📊 Current distribution:")
        for intent, count in distribution.items():
            print(f"  {intent}: {count}")
        return distribution

    def generate_samples_for_intent(self, intent: str, needed: int) -> List[Dict]:
        """Tạo samples cho intent cụ thể"""
        if intent not in self.templates:
            return []
        
        samples = []
        templates = self.templates[intent]
        
        for _ in range(needed):
            template = random.choice(templates)
            
            # Fill template với data phù hợp
            filled_text = self._fill_template(template, intent)
            
            if filled_text and filled_text not in [s['text'] for s in samples]:
                samples.append({
                    'text': filled_text,
                    'intent': intent,
                    'category': self._get_category(intent),
                    'confidence': 1.0
                })
                
        return samples

    def _fill_template(self, template: str, intent: str) -> str:
        """Fill template với data ngẫu nhiên"""
        import re
        
        # Tìm tất cả variables trong template
        variables = re.findall(r'\{(\w+)\}', template)
        
        if not variables:
            return template
            
        # Fill từng variable
        filled = template
        for var in variables:
            if var in self.template_data:
                value = random.choice(self.template_data[var])
                filled = filled.replace(f'{{{var}}}', value)
            else:
                # Nếu không có data cho variable này, return empty
                return ""
                
        return filled

    def _get_category(self, intent: str) -> str:
        """Map intent to category"""
        category_map = {
            'general_health': 'sức_khỏe_tổng_quát',
            'greeting': 'chào_hỏi',
            'drug_interaction': 'tương_tác_thuốc',
            'diet_lifestyle': 'chế_độ_sinh_hoạt',
            'medical_procedure': 'thủ_thuật_y_tế'
        }
        return category_map.get(intent, intent)

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cân bằng toàn bộ dataset"""
        print(f"\n🎯 Cân bằng dataset lên tối thiểu {self.min_samples} samples/intent")
        
        distribution = self.analyze_distribution(df)
        all_samples = df.to_dict('records')
        
        for intent, current_count in distribution.items():
            if current_count < self.min_samples:
                needed = self.min_samples - current_count
                print(f"🔄 {intent}: có {current_count}, cần thêm {needed}")
                
                new_samples = self.generate_samples_for_intent(intent, needed)
                
                if new_samples:
                    all_samples.extend(new_samples)
                    print(f"  ✅ Đã tạo {len(new_samples)} mẫu cho {intent}")
                else:
                    print(f"  ⚠️ Không thể tạo mẫu cho {intent}")
            else:
                print(f"✅ {intent}: {current_count} (đủ)")
        
        # Tạo DataFrame mới
        balanced_df = pd.DataFrame(all_samples)
        
        # Shuffle data
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return balanced_df

    def save_balanced_dataset(self, df: pd.DataFrame, output_path: str):
        """Lưu dataset đã cân bằng"""
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 Saved balanced dataset: {output_path}")
        
        # Final statistics
        print("\n📊 Final distribution:")
        final_dist = df['intent'].value_counts().to_dict()
        total = len(df)
        
        for intent, count in sorted(final_dist.items()):
            percentage = (count / total) * 100
            print(f"  {intent}: {count} ({percentage:.1f}%)")
        
        print(f"\n🎉 Total samples: {total}")

def main():
    print("⚖️ Dataset Balancer - Cân bằng hoàn toàn dataset")
    print("=" * 60)
    
    balancer = DatasetBalancer()
    
    # Load current dataset
    input_file = "data/medical_intent_training_dataset_5k_smart.csv"
    df = balancer.load_current_dataset(input_file)
    
    # Balance dataset
    balanced_df = balancer.balance_dataset(df)
    
    # Save balanced dataset
    output_file = "data/medical_intent_dataset_v4_balanced.csv"
    balancer.save_balanced_dataset(balanced_df, output_file)
    
    print(f"\n🚀 Dataset cân bằng hoàn tất!")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")

if __name__ == "__main__":
    main()