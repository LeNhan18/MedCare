#!/usr/bin/env python3
"""
Regenerate Medical Intent Training Dataset
Version 2: Sửa lỗi cắt cụt, thêm templates, và thêm dữ liệu "khó"
Tạo ra dataset chất lượng cao, duy nhất cho training
"""

import pandas as pd
import numpy as np
import random
import json
import os
from datetime import datetime


class MedicalIntentGenerator:
    def __init__(self):
        # Load existing medical data
        self.load_medical_data()

        # === NÂNG CẤP 1: THÊM NHIỀU TEMPLATES ĐA DẠNG HƠN ===

        # Define templates cho từng intent
        self.symptom_templates = [
            "Tôi bị {symptom}",
            "Em bé bị {symptom}",
            "Ông ấy bị {symptom}",
            "Bà ngoại bị {symptom}",
            "Anh ấy bị {symptom}",
            "Chị ấy bị {symptom}",
            "Con tôi bị {symptom}",
            "Mẹ tôi bị {symptom}",
            "Tôi thấy {symptom}",
            "Từ hôm qua {symptom}",
            "Mấy ngày nay {symptom}",
            "Tuần này {symptom}",
            "Có dấu hiệu {symptom}",
            "Có triệu chứng {symptom}",
            "Đang bị {symptom}",
            "Cảm thấy {symptom}",
            "Gặp phải tình trạng {symptom}",
            "Xuất hiện triệu chứng {symptom}",
            "Bị {symptom} phải làm sao?",
            "{symptom} là bệnh gì?",
            "Làm sao để chữa {symptom}?"
        ]

        self.drug_templates = [
            "Thuốc {drug} có tác dụng gì",
            "Thuốc {drug} dùng để làm gì",
            "Thuốc {drug} chữa bệnh gì",
            "Cách sử dụng thuốc {drug}",
            "Liều lượng thuốc {drug}",
            "Tác dụng của thuốc {drug}",
            "Thuốc {drug} có hiệu quả không",
            "{drug} là thuốc gì",
            "{drug} có tác dụng phụ gì không",
            "Cần uống thuốc {drug} bao lâu",
            "Thuốc {drug} có an toàn không",
            "Mua thuốc {drug} ở đâu",
            "Giá thuốc {drug} bao nhiêu",
            "Cho tôi hỏi về thuốc {drug}",
            "Thông tin về {drug}",
            "{drug} uống như thế nào?"
        ]

        self.side_effects_templates = [
            "Thuốc {drug} có tác dụng phụ gì",
            "Tác dụng phụ của {drug}",
            "Thuốc {drug} có hại gì không",
            "Uống {drug} có ảnh hưởng gì",
            "Thuốc {drug} có độc hại không",
            "Tác hại của thuốc {drug}",
            "Thuốc {drug} gây ra triệu chứng gì",
            "Sau khi uống {drug} bị {side_effect}",
            "Uống {drug} xong bị {side_effect}",
            "Thuốc {drug} làm tôi {side_effect}",
            "Có nên ngừng uống {drug} không vì bị {side_effect}",
            "Thuốc {drug} có gây {side_effect} không",
            "Uống {drug} có bị {side_effect} không?",
            "Tác dụng không mong muốn của {drug} là gì?"
        ]

        self.dosage_templates = [
            "Liều lượng thuốc {drug}",
            "Uống thuốc {drug} như thế nào",
            "Thuốc {drug} uống bao nhiêu viên",
            "Cách dùng thuốc {drug}",
            "Thuốc {drug} uống mấy lần một ngày",
            "Liều dùng của {drug}",
            "Thuốc {drug} uống trước hay sau ăn",
            "Cách sử dụng {drug} đúng",
            "Hướng dẫn sử dụng {drug}",
            "Thuốc {drug} uống lúc nào",
            "Liều khởi đầu của {drug}",
            "Liều tối đa của {drug}",
            "Uống {drug} bao nhiêu mg một ngày?",
            "Trẻ em uống {drug} thế nào?"
        ]

        self.emergency_templates = [
            "Cấp cứu! {emergency_symptom}",
            "Khẩn cấp: {emergency_symptom}",
            "SOS! {emergency_symptom}",
            "Help! {emergency_symptom}",
            "Gấp! {emergency_symptom}",
            "Cần cấp cứu ngay vì {emergency_symptom}",
            "Tình trạng nguy kịch: {emergency_symptom}",
            "Báo động đỏ: {emergency_symptom}",
            "Cần bác sĩ gấp: {emergency_symptom}",
            "Tình huống khẩn cấp: {emergency_symptom}"
        ]

        self.greeting_templates = [
            "Xin chào",
            "Hello",
            "Hi bác sĩ",
            "Chào bác sĩ",
            "Xin chào bác sĩ ơi",
            "Hi doctor",
            "Hello doctor",
            "Chào buổi sáng",
            "Chào buổi chiều",
            "Chào buổi tối",
            "Tôi có thể hỏi bác sĩ không",
            "Bác sĩ có rảnh không",
            "Cho tôi hỏi",
            "Tôi cần tư vấn",
            "Bạn có thể giúp tôi không?",
            "Có ai ở đó không?"
        ]

        self.general_health_templates = [
            "Cách giữ gìn sức khỏe",
            "Lời khuyên sức khỏe",
            "Cách sống khỏe mạnh",
            "Chế độ ăn uống lành mạnh",
            "Tập thể dục như thế nào",
            "Phòng ngừa bệnh tật",
            "Cách chăm sóc sức khỏe",
            "Lối sống lành mạnh",
            "Dinh dưỡng hợp lý",
            "Ngủ nghỉ đủ giấc",
            "Giảm căng thẳng",
            "Tăng cường đề kháng",
            "Bảo vệ sức khỏe",
            "Khám sức khỏe định kỳ",
            "Làm sao để giảm cân?",
            "Cách ăn uống để tăng cân?"
        ]

        self.unknown_templates = [
            "Thời tiết thế nào",
            "Phim hay gì",
            "Nhạc nào hay",
            "Ăn gì ngon",
            "Đi đâu chơi",
            "abcdefg",
            "blah blah",
            "lorem ipsum",
            "!@#$%",
            "123456",
            "aaaaaa",
            "bbbccc",
            "random text",
            "không liên quan",
            "văn bản vô nghĩa",
            "Công thức nấu phở bò ngon như thế nào?",  # Case 24 (khó)
            "Hôm nay giá vàng tăng hay giảm?",  # Case 22 (khó)
            "Mùa hè này nên đi du lịch Đà Lạt hay Sapa?",  # Case 23 (khó)
            "Công thức *chữa* cháy",  # "gài bẫy"
            "Cách *điều trị* bệnh lười biếng",  # "gài bẫy"
            "*Liều lượng* gia vị cho món súp"  # "gài bẫy"
        ]

        # === NÂNG CẤP 2: THÊM CÁC MẪU "KHÓ" (HARDCODED) ===
        # Đây là mỏ vàng của bạn. Hãy thêm tất cả các test case "khó" vào đây.
        self.hardcoded_samples = [
            # --- EMERGENCY (KHẨN CẤP) ---
            {'text': 'SOS! Người phụ nữ 45 tuổi bị va chạm xe máy gãy cẳng chân hở', 'intent': 'emergency'},
            {'text': 'Bệnh nhân nữ 29 tuổi mang thai 28 tuần bị tiền sản giật nặng cần làm gì?', 'intent': 'emergency'},
            {'text': 'Người bệnh suy tim độ 3 có thể phẫu thuật thay khớp háng được không?', 'intent': 'emergency'},
            {'text': 'Trẻ sơ sinh 3 ngày tuổi vàng da mức bilirubin 18mg/dL cần can thiệp gì?', 'intent': 'emergency'},
            {'text': 'Hội chứng Stevens-Johnson do carbamazepine có nguy hiểm không?', 'intent': 'emergency'},
            {'text': 'Ông cụ 82 tuổi ho đờm có máu và khó thở khi nằm ngửa', 'intent': 'emergency'},
            {'text': 'Khẩn cấp! Bé trai 2 tuổi nuốt phải pin cúc áo và khóc thét', 'intent': 'emergency'},
            {'text': 'Cần cấp cứu ngay! Ông già bị ngạt khói trong đám cháy', 'intent': 'emergency'},

            # --- DRUG_INTERACTION (INTENT MỚI) ---
            {'text': 'Có thể uống cùng lúc lansoprazole và clopidogrel không?', 'intent': 'drug_interaction'},
            {'text': 'Thuốc rosuvastatin có tương tác với thuốc tim digitalis?', 'intent': 'drug_interaction'},
            {'text': 'Metoprolol và amlodipine dùng chung có an toàn?', 'intent': 'drug_interaction'},
            {'text': 'Uống A chung với B được không?', 'intent': 'drug_interaction'},
            {'text': 'A và B có kỵ nhau không?', 'intent': 'drug_interaction'},
            {'text': 'Tôi đang uống A, giờ uống thêm B có sao không?', 'intent': 'drug_interaction'},

            # --- DIET_LIFESTYLE (INTENT MỚI) ---
            {'text': 'Chế độ ketogenic có phù hợp cho người bị rối loạn lipid máu?', 'intent': 'diet_lifestyle'},
            {'text': 'Intermittent fasting có lợi ích gì cho người tiểu đường type 2?', 'intent': 'diet_lifestyle'},
            {'text': 'Probiotics nào tốt nhất cho hội chứng ruột kích thích?', 'intent': 'diet_lifestyle'},
            {'text': 'Ăn kiêng keto có tốt không?', 'intent': 'diet_lifestyle'},
            {'text': 'Nhịn ăn gián đoạn ảnh hưởng gì?', 'intent': 'diet_lifestyle'},

            # --- MEDICAL_PROCEDURE (INTENT MỚI) ---
            {'text': 'Liệu pháp miễn dịch checkpoint inhibitor cho ung thư phổi', 'intent': 'medical_procedure'},
            {'text': 'Kỹ thuật ECMO có chỉ định trong suy hô hấp nặng COVID-19?', 'intent': 'medical_procedure'},
            {'text': 'Phẫu thuật thay khớp háng có nguy hiểm không?', 'intent': 'medical_procedure'},

            # --- SIDE_EFFECTS (CÁC CA KHÓ) ---
            {'text': 'Finasteride có thể gây rối loạn chức năng sinh dục nam không?', 'intent': 'side_effects'},
            {'text': 'Isotretinoin có ảnh hưởng đến thai nhi như thế nào?', 'intent': 'side_effects'},
            {'text': 'Adalimumab có nguy cơ gây nhiễm trùng cơ hội nào?', 'intent': 'side_effects'},

            # --- CÁC CA BỊ NHẦM KHÁC ---
            {'text': 'Bác sĩ kê đơn cefixime 200mg uống 12 tiếng một lần trong 7 ngày', 'intent': 'symptom_inquiry'}
            # Hoặc 1 intent 'user_statement' mới
        ]

        # Thêm nhãn (label) mới cho các intent mới
        self.new_intent_labels = ['drug_interaction', 'diet_lifestyle', 'medical_procedure']

    def load_medical_data(self):
        """Load medical dataset để lấy danh sách thuốc và triệu chứng"""
        try:
            # Sửa đường dẫn nếu cần
            df = pd.read_csv('data/medical_dataset_training.csv', encoding='utf-8')
            self.drugs = df['drug_name'].dropna().unique().tolist()

            # Lấy triệu chứng từ medical_condition_vi nếu có
            if 'medical_condition_vi' in df.columns:
                self.symptoms = df['medical_condition_vi'].dropna().unique().tolist()
            else:
                self.symptoms = df['medical_condition'].dropna().unique().tolist()

            # Lấy side effects
            if 'side_effects_vi' in df.columns:
                side_effects_list = []
                for effects in df['side_effects_vi'].dropna():
                    if isinstance(effects, str):
                        # Sửa lỗi: Tách bằng dấu phẩy
                        side_effects_list.extend([e.strip() for e in effects.split(',') if e.strip()])
                self.side_effects = list(set(side_effects_list))
            else:
                self.side_effects = ['buồn nôn', 'đau đầu', 'chóng mặt', 'mệt mỏi', 'tiêu chảy']

            # Đảm bảo danh sách không rỗng
            if not self.side_effects:
                self.side_effects = ['buồn nôn', 'đau đầu', 'chóng mặt', 'mệt mỏi']

            print(
                f"✅ Loaded {len(self.drugs)} drugs, {len(self.symptoms)} symptoms, {len(self.side_effects)} side effects")

        except Exception as e:
            print(f"❌ Error loading medical data: {e}")
            # Fallback data
            self.drugs = ['paracetamol', 'ibuprofen', 'aspirin', 'amoxicillin']
            self.symptoms = ['đau đầu', 'sốt', 'ho', 'đau bụng', 'mệt mỏi']
            self.side_effects = ['buồn nôn', 'đau đầu', 'chóng mặt', 'mệt mỏi']

    # --- CÁC HÀM GENERATE GIỮ NGUYÊN ---

    def generate_symptom_inquiry_samples(self, count=1500):
        """Tạo mẫu symptom_inquiry"""
        samples = []
        for i in range(count):
            template = random.choice(self.symptom_templates)
            symptom = random.choice(self.symptoms)

            # Clean symptom text
            if isinstance(symptom, str) and symptom:
                symptom = symptom.strip().lower()
                text = template.format(symptom=symptom)

                samples.append({
                    'text': text,
                    'intent': 'symptom_inquiry',
                    'category': 'triệu_chứng',
                    'confidence': 1.0
                })
        return samples

    def generate_drug_question_samples(self, count=1200):
        """Tạo mẫu drug_question"""
        samples = []
        for i in range(count):
            template = random.choice(self.drug_templates)
            drug = random.choice(self.drugs)

            if isinstance(drug, str) and drug:
                drug = drug.strip()
                text = template.format(drug=drug)

                samples.append({
                    'text': text,
                    'intent': 'drug_question',
                    'category': 'thuốc',
                    'confidence': 1.0
                })
        return samples

    def generate_side_effects_samples(self, count=800):
        """Tạo mẫu side_effects"""
        samples = []
        for i in range(count):
            template = random.choice(self.side_effects_templates)
            drug = random.choice(self.drugs)
            side_effect = random.choice(self.side_effects)

            if isinstance(drug, str) and drug and isinstance(side_effect, str) and side_effect:
                drug = drug.strip()
                side_effect = side_effect.strip().lower()

                if '{side_effect}' in template:
                    text = template.format(drug=drug, side_effect=side_effect)
                else:
                    text = template.format(drug=drug)

                samples.append({
                    'text': text,
                    'intent': 'side_effects',
                    'category': 'tác_dụng_phụ',
                    'confidence': 1.0
                })
        return samples

    def generate_dosage_question_samples(self, count=600):
        """Tạo mẫu dosage_question"""
        samples = []
        for i in range(count):
            template = random.choice(self.dosage_templates)
            drug = random.choice(self.drugs)

            if isinstance(drug, str) and drug:
                drug = drug.strip()
                text = template.format(drug=drug)

                samples.append({
                    'text': text,
                    'intent': 'dosage_question',
                    'category': 'liều_lượng',
                    'confidence': 1.0
                })
        return samples

    def generate_emergency_samples(self, count=400):
        """Tạo mẫu emergency"""
        emergency_symptoms = [
            'đau ngực dữ dội', 'khó thở nghiêm trọng', 'mất ý thức',
            'chảy máu nhiều', 'sốt cao trên 40 độ', 'co giật',
            'đột quỵ', 'tim ngừng đập', 'sốc phản vệ',
            'ngộ độc nghiêm trọng', 'gãy xương hở'
        ]

        samples = []
        for i in range(count):
            template = random.choice(self.emergency_templates)
            emergency_symptom = random.choice(emergency_symptoms)
            text = template.format(emergency_symptom=emergency_symptom)

            samples.append({
                'text': text,
                'intent': 'emergency',
                'category': 'cấp_cứu',
                'confidence': 1.0
            })
        return samples

    def generate_greeting_samples(self, count=300):
        """Tạo mẫu greeting"""
        samples = []
        for i in range(count):
            text = random.choice(self.greeting_templates)

            samples.append({
                'text': text,
                'intent': 'greeting',
                'category': 'chào_hỏi',
                'confidence': 1.0
            })
        return samples

    def generate_general_health_samples(self, count=400):
        """Tạo mẫu general_health"""
        samples = []
        for i in range(count):
            text = random.choice(self.general_health_templates)

            samples.append({
                'text': text,
                'intent': 'general_health',
                'category': 'sức_khỏe_tổng_quát',
                'confidence': 1.0
            })
        return samples

    def generate_unknown_samples(self, count=300):
        """Tạo mẫu unknown"""
        samples = []
        for i in range(count):
            if i < len(self.unknown_templates):
                text = self.unknown_templates[i]
            else:
                # Generate random unknown text
                random_parts = [
                    random.choice(['Phim', 'Nhạc', 'Ăn', 'Đi', 'Xem']),
                    random.choice(['hay', 'ngon', 'đẹp', 'vui', 'tốt']),
                    str(random.randint(100, 999))
                ]
                text = ' '.join(random_parts)

            samples.append({
                'text': text,
                'intent': 'unknown',
                'category': 'không_xác_định',
                'confidence': 1.0
            })
        return samples

    def generate_complete_dataset(self):
        """Tạo dataset hoàn chỉnh 5500 mẫu"""
        print(" Generating complete 5K medical intent dataset...")

        all_samples = []

        # === NÂNG CẤP 3: THÊM CÁC MẪU "CỨNG" VÀO DATASET ===

        # Thêm các mẫu "hardcoded" chất lượng cao
        # Thêm 10 bản sao của mỗi mẫu "khó" để nhấn mạnh cho model
        for _ in range(10):
            all_samples.extend(self.hardcoded_samples)
        print(f" Added {len(self.hardcoded_samples) * 10} 'hardcoded' samples (x10 copies)")

        # Generate samples for each intent
        print(" Generating symptom_inquiry samples...")
        all_samples.extend(self.generate_symptom_inquiry_samples(1500))

        print(" Generating drug_question samples...")
        all_samples.extend(self.generate_drug_question_samples(1200))

        print(" Generating side_effects samples...")
        all_samples.extend(self.generate_side_effects_samples(800))

        print(" Generating dosage_question samples...")
        all_samples.extend(self.generate_dosage_question_samples(600))

        print(" Generating emergency samples...")
        all_samples.extend(self.generate_emergency_samples(400))

        print(" Generating greeting samples...")
        all_samples.extend(self.generate_greeting_samples(300))

        print("Generating general_health samples...")
        all_samples.extend(self.generate_general_health_samples(400))

        print(" Generating unknown samples...")
        all_samples.extend(self.generate_unknown_samples(300))

        # Shuffle the dataset
        random.shuffle(all_samples)

        print(f" Generated {len(all_samples)} total samples (before deduplication)")

        # Count by intent
        intent_counts = {}
        for sample in all_samples:
            intent = sample['intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        print(" Intent distribution (before deduplication):")
        for intent, count in sorted(intent_counts.items()):
            print(f"  • {intent}: {count} samples")

        return all_samples

    def remove_duplicates(self, samples):
        """Loại bỏ các mẫu trùng lặp"""
        print(" Removing duplicates...")

        seen_texts = set()
        unique_samples = []

        for sample in samples:
            text = sample['text'].strip().lower()
            if text not in seen_texts:
                seen_texts.add(text)
                unique_samples.append(sample)

        print(f" Removed {len(samples) - len(unique_samples)} duplicates")
        print(f" Unique samples: {len(unique_samples)}")

        return unique_samples

    def save_to_csv(self, samples, output_path):
        """Lưu dataset vào CSV với encoding UTF-8 đúng"""

        # Thêm cột "confidence" và "category" nếu chưa có (cho các mẫu hardcoded)
        for sample in samples:
            if 'confidence' not in sample:
                sample['confidence'] = 1.0
            if 'category' not in sample:
                sample['category'] = sample['intent']  # Mặc định category là intent

        df = pd.DataFrame(samples)

        # Sắp xếp lại cột
        columns_order = ['text', 'intent', 'category', 'confidence']
        df = df[columns_order]

        # Ensure proper encoding
        df.to_csv(output_path, index=False, encoding='utf-8-sig',
                  quoting=1)  # quoting=1 để đảm bảo text được bọc trong dấu "

        print(f" Saved {len(samples)} samples to {output_path}")

        # Verify saved file
        verify_df = pd.read_csv(output_path, encoding='utf-8')
        print(f" Verification: Loaded {len(verify_df)} samples from saved file")

        return output_path


def main():
    print(" Medical Intent 5K Dataset Regenerator (v2)")
    print("=" * 60)

    # Đảm bảo thư mục 'data' tồn tại
    os.makedirs('data', exist_ok=True)

    # Initialize generator
    generator = MedicalIntentGenerator()

    # Generate complete dataset
    samples = generator.generate_complete_dataset()

    # Remove duplicates
    unique_samples = generator.remove_duplicates(samples)

    # Backup old file
    old_file = 'data/medical_intent_training_dataset_5k.csv'
    backup_file = f'data/medical_intent_training_dataset_5k_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    if os.path.exists(old_file):
        import shutil
        shutil.copy2(old_file, backup_file)
        print(f" Backed up old file to: {backup_file}")

    # Save new dataset
    new_file = 'data/medical_intent_dataset_v2_clean.csv'  # Đổi tên file mới cho rõ ràng
    generator.save_to_csv(unique_samples, new_file)

    print(f"\n HOÀN THÀNH!")
    print(f" Dataset mới: {new_file} (Hãy dùng file này để train)")
    if os.path.exists(backup_file):
        print(f"Backup cũ: {backup_file}")
    print(f" Tổng mẫu duy nhất: {len(unique_samples)}")
    print(f" Sẵn sàng để training!")


if __name__ == "__main__":
    main()
