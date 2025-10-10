#!/usr/bin/env python3
"""
Phân tích và tạo dataset training cho Medical Chatbot
Ghép dữ liệu từ các file đã dịch và tạo dataset hoàn chỉnh
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_translation_files():
    """Phân tích các file dịch để xem file nào có đủ cột dịch"""
    
    print("🔍 Phân tích các file dữ liệu đã dịch...")
    print("=" * 50)
    
    files_to_check = [
        "data/drugs_data_translated_full.csv",
        "data/drugs_data_translated_partial_medical_condition.csv", 
        "data/drugs_data_translated_partial_side_effects.csv",
        "data/drugs_data_translated_partial_drug_classes.csv",
        "data/drugs_data_translated_partial_medical_condition_description.csv",
        "data/drugs_data_simple_translated.csv"
    ]
    
    analysis_results = {}
    
    for file_path in files_to_check:
        try:
            if Path(file_path).exists():
                df = pd.read_csv(file_path, nrows=5)  # Chỉ đọc 5 dòng đầu để kiểm tra
                
                # Tìm các cột tiếng Việt
                vi_columns = [col for col in df.columns if col.endswith('_vi')]
                
                analysis_results[file_path] = {
                    'exists': True,
                    'total_columns': len(df.columns),
                    'vietnamese_columns': vi_columns,
                    'num_vi_columns': len(vi_columns),
                    'file_size': Path(file_path).stat().st_size / 1024 / 1024  # MB
                }
                
                print(f"📄 {file_path}")
                print(f"   ✅ Tổng số cột: {len(df.columns)}")
                print(f"   🇻🇳 Cột tiếng Việt: {len(vi_columns)} - {vi_columns}")
                print(f"   📊 Kích thước: {analysis_results[file_path]['file_size']:.2f} MB")
                print()
            else:
                analysis_results[file_path] = {'exists': False}
                print(f"❌ {file_path} - Không tồn tại")
                
        except Exception as e:
            analysis_results[file_path] = {'exists': False, 'error': str(e)}
            print(f"⚠️ {file_path} - Lỗi: {e}")
    
    return analysis_results

def create_training_dataset():
    """Tạo dataset training từ file dịch tốt nhất"""
    
    print("\n🎯 Tạo dataset training...")
    print("=" * 50)
    
    # Thử đọc file full trước
    try:
        df_full = pd.read_csv("data/drugs_data_translated_full.csv")
        print(f"✅ Đọc được file full với {len(df_full)} dòng")
        
        # Kiểm tra các cột tiếng Việt
        vi_columns = [col for col in df_full.columns if col.endswith('_vi')]
        
        if len(vi_columns) >= 4:  # Cần ít nhất 4 cột đã dịch
            print(f"🇻🇳 Có {len(vi_columns)} cột tiếng Việt: {vi_columns}")
            base_df = df_full.copy()
        else:
            print("⚠️ File full chưa có đủ cột dịch, sẽ ghép từ các file partial...")
            base_df = merge_partial_files()
            
    except Exception as e:
        print(f"❌ Không đọc được file full: {e}")
        print("🔄 Sẽ ghép từ các file partial...")
        base_df = merge_partial_files()
    
    # Làm sạch và chuẩn hóa dữ liệu
    clean_df = clean_training_data(base_df)
    
    # Tạo các format khác nhau cho training
    create_training_formats(clean_df)
    
    return clean_df

def merge_partial_files():
    """Ghép dữ liệu từ các file partial"""
    
    print("🔄 Ghép dữ liệu từ các file partial...")
    
    # Bắt đầu với file gốc UTF-8
    df = pd.read_csv("data/drugs_data_utf8.csv")
    print(f"📂 Base dataset: {len(df)} dòng")
    
    # Ghép medical_condition_vi
    try:
        df_mc = pd.read_csv("data/drugs_data_translated_partial_medical_condition.csv")
        if 'medical_condition_vi' in df_mc.columns:
            df = df.merge(df_mc[['drug_name', 'medical_condition_vi']], on='drug_name', how='left')
            print("✅ Ghép medical_condition_vi")
    except:
        print("❌ Không ghép được medical_condition_vi")
    
    # Ghép side_effects_vi
    try:
        df_se = pd.read_csv("data/drugs_data_translated_partial_side_effects.csv")
        if 'side_effects_vi' in df_se.columns:
            df = df.merge(df_se[['drug_name', 'side_effects_vi']], on='drug_name', how='left')
            print("✅ Ghép side_effects_vi")
    except:
        print("❌ Không ghép được side_effects_vi")
    
    # Ghép drug_classes_vi
    try:
        df_dc = pd.read_csv("data/drugs_data_translated_partial_drug_classes.csv")
        if 'drug_classes_vi' in df_dc.columns:
            df = df.merge(df_dc[['drug_name', 'drug_classes_vi']], on='drug_name', how='left')
            print("✅ Ghép drug_classes_vi")
    except:
        print("❌ Không ghép được drug_classes_vi")
    
    # Ghép medical_condition_description_vi
    try:
        df_mcd = pd.read_csv("data/drugs_data_translated_partial_medical_condition_description.csv")
        if 'medical_condition_description_vi' in df_mcd.columns:
            df = df.merge(df_mcd[['drug_name', 'medical_condition_description_vi']], on='drug_name', how='left')
            print("✅ Ghép medical_condition_description_vi")
    except:
        print("❌ Không ghép được medical_condition_description_vi")
    
    print(f"🎯 Dataset sau khi ghép: {len(df)} dòng")
    return df

def clean_training_data(df):
    """Làm sạch dữ liệu cho training"""
    
    print("\n🧹 Làm sạch dữ liệu...")
    print("=" * 50)
    
    # Loại bỏ các cột Unnamed không cần thiết
    df_clean = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    print(f"🗑️ Loại bỏ các cột Unnamed: {len(df.columns)} -> {len(df_clean.columns)} cột")
    
    # Chỉ giữ các cột cần thiết cho training
    essential_columns = [
        'drug_name', 'medical_condition', 'side_effects', 'generic_name', 
        'drug_classes', 'brand_names', 'rx_otc', 'medical_condition_description',
        'rating', 'no_of_reviews'
    ]
    
    # Thêm các cột tiếng Việt nếu có
    vi_columns = [col for col in df_clean.columns if col.endswith('_vi')]
    essential_columns.extend(vi_columns)
    
    # Chỉ giữ các cột tồn tại
    available_columns = [col for col in essential_columns if col in df_clean.columns]
    df_clean = df_clean[available_columns]
    
    print(f"📋 Các cột được giữ lại: {available_columns}")
    
    # Loại bỏ các dòng có drug_name trống
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['drug_name'])
    print(f"🔍 Loại bỏ dòng trống drug_name: {initial_rows} -> {len(df_clean)} dòng")
    
    # Loại bỏ duplicate
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['drug_name'])
    print(f"🔄 Loại bỏ duplicate: {initial_rows} -> {len(df_clean)} dòng")
    
    print(f"\n✅ Dataset sạch: {len(df_clean)} dòng, {len(df_clean.columns)} cột")
    
    # Hiển thị thống kê dịch
    for col in vi_columns:
        if col in df_clean.columns:
            translated_count = df_clean[col].notna().sum()
            total_count = len(df_clean)
            percentage = (translated_count / total_count) * 100
            print(f"🇻🇳 {col}: {translated_count}/{total_count} ({percentage:.1f}%) đã dịch")
    
    return df_clean

def create_training_formats(df):
    """Tạo các format khác nhau cho training model"""
    
    print(f"\n📝 Tạo các format training...")
    print("=" * 50)
    
    # 1. Tạo file CSV sạch cho training chính
    output_file = "data/medical_dataset_training.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"💾 Lưu dataset chính: {output_file}")
    
    # 2. Tạo JSON format cho chatbot
    json_data = []
    for _, row in df.iterrows():
        if pd.notna(row.get('medical_condition_vi')):
            entry = {
                "drug_name": row['drug_name'],
                "condition_en": row.get('medical_condition', ''),
                "condition_vi": row.get('medical_condition_vi', ''),
                "side_effects_en": row.get('side_effects', ''),
                "side_effects_vi": row.get('side_effects_vi', ''),
                "drug_class_en": row.get('drug_classes', ''),
                "drug_class_vi": row.get('drug_classes_vi', ''),
                "description_en": row.get('medical_condition_description', ''),
                "description_vi": row.get('medical_condition_description_vi', ''),
                "rx_otc": row.get('rx_otc', ''),
                "rating": row.get('rating', 0)
            }
            json_data.append(entry)
    
    json_file = "data/medical_dataset_training.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Lưu dataset JSON: {json_file} ({len(json_data)} entries)")
    
    # 3. Tạo file chỉ symptoms-drugs cho NER
    symptom_data = []
    for _, row in df.iterrows():
        if pd.notna(row.get('medical_condition_vi')):
            symptom_data.append({
                "symptom_vi": row.get('medical_condition_vi', ''),
                "symptom_en": row.get('medical_condition', ''),
                "recommended_drug": row['drug_name'],
                "drug_class": row.get('drug_classes_vi', row.get('drug_classes', '')),
                "rx_otc": row.get('rx_otc', ''),
                "rating": row.get('rating', 0)
            })
    
    symptom_file = "data/symptom_drug_mapping.json"
    with open(symptom_file, 'w', encoding='utf-8') as f:
        json.dump(symptom_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Lưu symptom-drug mapping: {symptom_file} ({len(symptom_data)} entries)")
    
    return {
        'main_dataset': output_file,
        'json_dataset': json_file,
        'symptom_mapping': symptom_file,
        'total_drugs': len(df),
        'translated_entries': len(json_data)
    }

def main():
    print("🏥 Medical Dataset Preparation for Training")
    print("=" * 60)
    
    # Phân tích các file
    analysis = analyze_translation_files()
    
    # Tạo dataset training
    training_data = create_training_dataset()
    
    # Tạo summary report
    print(f"\n📊 BÁO CÁO TỔNG KẾT")
    print("=" * 60)
    print(f"✅ Dataset training đã sẵn sàng!")
    print(f"📁 File chính: data/medical_dataset_training.csv")
    print(f"📄 File JSON: data/medical_dataset_training.json")
    print(f"🎯 File symptom mapping: data/symptom_drug_mapping.json")
    print(f"💊 Tổng số thuốc: {len(training_data)}")
    
    # Kiểm tra các cột tiếng Việt
    vi_columns = [col for col in training_data.columns if col.endswith('_vi')]
    print(f"🇻🇳 Các cột đã dịch: {vi_columns}")
    
    for col in vi_columns:
        translated_count = training_data[col].notna().sum()
        total_count = len(training_data)
        percentage = (translated_count / total_count) * 100
        print(f"   - {col}: {translated_count}/{total_count} ({percentage:.1f}%)")
    
    print(f"\n🚀 Có thể bắt đầu training chatbot với dữ liệu này!")

if __name__ == "__main__":
    main()