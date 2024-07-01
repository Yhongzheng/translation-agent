import os

import translation_agent as ta


if __name__ == "__main__":
    source_lang, target_lang, country = "English", "Chinese", "China"

    relative_path = r"C:\Users\yongjie.yang\Desktop\ceshi.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    full_path = os.path.join(script_dir, relative_path)

    with open(full_path, encoding="utf-8") as file:
        source_text = file.read()

    # print(f"Source text:\n\n{source_text}\n------------\n")

    translation = ta.translate(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        country=country,
    )

    # print(f"Translation:\n\n{translation}")

    # 保存翻译后的文本到源文件的地址
    translated_file_path = os.path.splitext(full_path)[0] + "_translated.txt"
    with open(translated_file_path, "w", encoding="utf-8") as file:
        file.write(translation)

    print(f"Translation saved to: {translated_file_path}")
