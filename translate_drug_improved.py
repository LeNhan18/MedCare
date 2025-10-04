import asyncio
import pandas as pd
from googletrans import Translator
import time
import sys

translator = Translator()

async def translate_text(text, delay=0.1):
    """Dịch văn bản với delay để tránh rate limit"""
    text = str(text)
    if text.strip().lower() in ("nan", "", "null") or pd.isna(text):
        return text
    
    try:
        # Thêm delay để tránh rate limit
        await asyncio.sleep(delay)
        
        # Giới hạn độ dài văn bản để tránh lỗi
        if len(text) > 5000:
            text = text[:5000] + "..."
        
        # Chạy translate trong thread để không block event loop
        translated = await asyncio.to_thread(
            translator.translate, text, src="en", dest="vi"
        )
        return translated.text
    except Exception as e:
        print(f"Lỗi dịch: {e}")
        return f"[ERROR: {str(e)}]"

async def translate_column_batch(df, column, batch_size=10):
    """Dịch cột theo batch để tránh overload"""
    print(f"Đang dịch cột: {column} ({len(df)} dòng)...")
    
    total_rows = len(df)
    translated_texts = []
    
    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        batch = df[column].iloc[i:batch_end]
        
        print(f"Đang dịch batch {i//batch_size + 1} ({i+1}-{batch_end}/{total_rows})...")
        
        # Dịch batch với delay lớn hơn
        batch_tasks = [translate_text(text, delay=0.5) for text in batch]
        batch_results = await asyncio.gather(*batch_tasks)
        
        translated_texts.extend(batch_results)
        
        # Pause giữa các batch
        if batch_end < total_rows:
            print("Đang nghỉ 2 giây...")
            await asyncio.sleep(2)
    
    df[column] = translated_texts
    return df

async def main():
    try:
        print("Đang đọc file CSV...")
        df = pd.read_csv("data/drugs_data.csv", encoding="cp1252")
        print(f"Đã đọc {len(df)} dòng dữ liệu")
        
        # Kiểm tra các cột có sẵn
        print("Các cột có trong file:")
        print(df.columns.tolist())
        
        # Chỉ dịch một số cột quan trọng và ngắn
        available_columns = []
        columns_to_translate = ['drug_name', 'medical_condition', 'generic_name']
        
        for col in columns_to_translate:
            if col in df.columns:
                available_columns.append(col)
                print(f"Sẽ dịch cột: {col}")
            else:
                print(f"Cột {col} không tồn tại")
        
        if not available_columns:
            print("Không có cột nào để dịch!")
            return
        
        # Giới hạn số dòng để test (chỉ dịch 50 dòng đầu)
        test_df = df.head(50).copy()
        print(f"Dịch thử {len(test_df)} dòng đầu...")
        
        for col in available_columns:
            test_df = await translate_column_batch(test_df, col, batch_size=5)
            print(f"Hoàn thành dịch cột {col}")
        
        # Lưu kết quả
        output_file = "data/drugs_data_translated_test.csv"
        test_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Đã lưu file dịch vào: {output_file}")
        
    except Exception as e:
        print(f"Lỗi chính: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())