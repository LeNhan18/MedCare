import pandas as pd
import time
import random
from googletrans import Translator

def translate_text_safe(text, translator, max_retries=3):
    """Dịch văn bản an toàn với retry và delay"""
    text = str(text)
    
    # Bỏ qua các giá trị rỗng
    if text.strip().lower() in ("nan", "", "null") or pd.isna(text):
        return text
    
    # Giới hạn độ dài
    if len(text) > 3000:
        text = text[:3000] + "..."
    
    for attempt in range(max_retries):
        try:
            # Random delay để tránh rate limit
            delay = random.uniform(1, 3)
            time.sleep(delay)
            
            # Tạo translator mới cho mỗi lần thử
            if attempt > 0:
                translator = Translator()
                time.sleep(2)  # Delay thêm khi retry
            
            result = translator.translate(text, src='en', dest='vi')
            return result.text
            
        except Exception as e:
            print(f"Lỗi dịch (lần thử {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Tăng thời gian chờ
                print(f"Đợi {wait_time} giây trước khi thử lại...")
                time.sleep(wait_time)
            else:
                return f"[FAILED: {text[:50]}...]"
    
    return text  # Trả về text gốc nếu không dịch được

def translate_column_safe(df, column_name, batch_size=5):
    """Dịch cột một cách an toàn"""
    print(f"\nĐang dịch cột: {column_name}")
    print(f"Tổng số dòng: {len(df)}")
    
    translator = Translator()
    translated_data = []
    
    for i, text in enumerate(df[column_name]):
        print(f"Đang dịch dòng {i+1}/{len(df)}: {str(text)[:50]}...")
        
        translated = translate_text_safe(text, translator)
        translated_data.append(translated)
        
        # Nghỉ sau mỗi batch
        if (i + 1) % batch_size == 0:
            print(f"Đã hoàn thành {i+1} dòng. Nghỉ 5 giây...")
            time.sleep(5)
    
    df[column_name + '_vi'] = translated_data
    return df

def main():
    try:
        print("Đang đọc file CSV...")
        # Chỉ đọc các cột cần thiết để tránh lỗi
        useful_columns = ['drug_name', 'medical_condition', 'side_effects', 'generic_name', 
                         'drug_classes', 'brand_names', 'medical_condition_description']
        
        df = pd.read_csv("data/drugs_data.csv", encoding="cp1252", usecols=useful_columns)
        print(f"Đã đọc {len(df)} dòng dữ liệu")
        print("Các cột đã đọc:", df.columns.tolist())
        
        # Test với 10 dòng đầu
        test_df = df.head(10).copy()
        print(f"\nTest với {len(test_df)} dòng đầu...")
        
        # Dịch các cột quan trọng
        columns_to_translate = ['drug_name', 'medical_condition', 'generic_name']
        
        for col in columns_to_translate:
            if col in test_df.columns:
                print(f"\n{'='*50}")
                print(f"BẮT ĐẦU DỊCH CỘT: {col}")
                print(f"{'='*50}")
                test_df = translate_column_safe(test_df, col, batch_size=3)
                print(f"Hoàn thành dịch cột: {col}")
                
                # Nghỉ lâu giữa các cột
                print("Nghỉ 10 giây trước khi dịch cột tiếp theo...")
                time.sleep(10)
        
        # Lưu kết quả
        output_file = "data/drugs_data_translated_fixed.csv"
        test_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nĐã lưu file dịch vào: {output_file}")
        
        # Hiển thị một số kết quả
        print("\nMột số kết quả dịch:")
        for col in columns_to_translate:
            if col in test_df.columns:
                print(f"\n{col.upper()}:")
                for i in range(min(3, len(test_df))):
                    original = test_df[col].iloc[i]
                    translated = test_df[col + '_vi'].iloc[i]
                    print(f"  Gốc: {original}")
                    print(f"  Dịch: {translated}")
                    print("-" * 40)
        
    except Exception as e:
        print(f"Lỗi chính: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("CHƯƠNG TRÌNH DỊCH THUỐC - PHIÊN BẢN ỐN ĐỊNH")
    print("=" * 60)
    main()