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
            delay = random.uniform(2, 5)  # Tăng delay
            time.sleep(delay)
            
            # Tạo translator mới cho mỗi lần thử
            if attempt > 0:
                translator = Translator()
                time.sleep(5)  # Delay thêm khi retry
            
            result = translator.translate(text, src='en', dest='vi')
            return result.text
            
        except Exception as e:
            print(f"Lỗi dịch (lần thử {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Tăng thời gian chờ
                print(f"Đợi {wait_time} giây trước khi thử lại...")
                time.sleep(wait_time)
            else:
                return f"[FAILED: {text[:50]}...]"
    
    return text  # Trả về text gốc nếu không dịch được

def translate_full_dataset():
    """Dịch toàn bộ dataset với chiến lược batch"""
    print("DỊCH TOÀN BỘ DATASET - PHIÊN BẢN ĐẦY ĐỦ")
    print("=" * 60)
    
    try:
        # Đọc dữ liệu
        print("Đang đọc file CSV...")
        useful_columns = ['drug_name', 'medical_condition', 'side_effects', 'generic_name', 
                         'drug_classes', 'brand_names', 'medical_condition_description']
        
        df = pd.read_csv("data/drugs_data.csv", encoding="cp1252", usecols=useful_columns)
        print(f"Đã đọc {len(df)} dòng dữ liệu")
        
        # Chọn cột để dịch (chỉ dịch những cột quan trọng)
        columns_to_translate = ['drug_name', 'medical_condition', 'generic_name', 'drug_classes']
        
        translator = Translator()
        
        # Dịch từng cột
        for col_idx, column in enumerate(columns_to_translate):
            print(f"\n{'='*60}")
            print(f"ĐANG DỊCH CỘT {col_idx+1}/{len(columns_to_translate)}: {column}")
            print(f"{'='*60}")
            
            translated_data = []
            total_rows = len(df)
            
            # Dịch từng dòng với progress
            for i, text in enumerate(df[column]):
                progress = (i + 1) / total_rows * 100
                print(f"[{progress:.1f}%] Dòng {i+1}/{total_rows}: {str(text)[:50]}...")
                
                translated = translate_text_safe(text, translator)
                translated_data.append(translated)
                
                # Lưu progress mỗi 50 dòng
                if (i + 1) % 50 == 0:
                    temp_df = df.copy()
                    temp_df[column + '_vi'] = translated_data + [''] * (len(df) - len(translated_data))
                    temp_df.to_csv(f"data/progress_backup_{column}_{i+1}.csv", 
                                 index=False, encoding="utf-8-sig")
                    print(f"✅ Đã backup progress tại dòng {i+1}")
                    
                    # Nghỉ lâu sau mỗi 50 dòng
                    print("💤 Nghỉ 30 giây để tránh rate limit...")
                    time.sleep(30)
            
            # Thêm cột đã dịch vào dataframe
            df[column + '_vi'] = translated_data
            
            # Lưu kết quả từng cột
            output_file = f"data/drugs_translated_column_{column}.csv"
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"✅ Đã lưu kết quả cột {column} vào: {output_file}")
            
            # Nghỉ lâu giữa các cột
            if col_idx < len(columns_to_translate) - 1:
                print("💤 Nghỉ 60 giây trước khi dịch cột tiếp theo...")
                time.sleep(60)
        
        # Lưu kết quả cuối cùng
        final_output = "data/drugs_data_fully_translated.csv"
        df.to_csv(final_output, index=False, encoding="utf-8-sig")
        print(f"\n🎉 HOÀN THÀNH! Đã lưu file cuối cùng: {final_output}")
        
        # Thống kê kết quả
        print(f"\n📊 THỐNG KÊ:")
        print(f"- Tổng số dòng: {len(df)}")
        print(f"- Số cột đã dịch: {len(columns_to_translate)}")
        for col in columns_to_translate:
            translated_col = col + '_vi'
            if translated_col in df.columns:
                failed_count = df[translated_col].str.contains('[FAILED:', na=False).sum()
                success_rate = (len(df) - failed_count) / len(df) * 100
                print(f"- {col}: {success_rate:.1f}% thành công")
        
    except Exception as e:
        print(f"❌ Lỗi chính: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Hỏi người dùng xác nhận
    print("⚠️  CẢNH BÁO: Việc dịch toàn bộ dataset sẽ mất NHIỀU GIỜ!")
    print("📋 Sẽ dịch 4 cột quan trọng cho 2966 dòng dữ liệu")
    print("💾 Sẽ tự động backup progress mỗi 50 dòng")
    print("⏰ Ước tính: 6-8 tiếng (do rate limiting)")
    
    choice = input("\n❓ Bạn có muốn tiếp tục? (y/N): ").lower().strip()
    if choice == 'y' or choice == 'yes':
        translate_full_dataset()
    else:
        print("❌ Đã hủy. Sử dụng script test nhỏ để thử nghiệm trước.")