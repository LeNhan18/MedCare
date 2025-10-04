import asyncio
import pandas as pd
from googletrans import Translator

translator = Translator()

async def translate_text(text):
    text = str(text)
    if text.strip().lower() != "nan":
        try:
            # Chạy translate trong thread để không block event loop
            translated = await asyncio.to_thread(
                translator.translate, text, src="en", dest="vi"
            )
            return translated.text
        except Exception as e:
            return f"[ERROR: {e}]"
    return text

async def translate_column(df, column):
    print(f"Đang dịch cột: {column} ...")
    tasks = [translate_text(text) for text in df[column]]
    df[column] = await asyncio.gather(*tasks)
    return df

async def main():
    df = pd.read_csv("data/drugs_data.csv", encoding="cp1252")
    columns_to_translate = ['generic_name', 'description']

    for col in columns_to_translate:
        df = await translate_column(df, col)

    df.to_csv("data/drugs_data_translated.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    asyncio.run(main())
