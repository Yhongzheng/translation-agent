import re

def merge_subtitles(subtitles):
    merged_subs = []
    previous_sub = None
    for sub in subtitles:
        if previous_sub and sub['text'] == previous_sub['text']:
            previous_sub['end'] = sub['end']
        else:
            if previous_sub:
                merged_subs.append(previous_sub)
            previous_sub = sub
    if previous_sub:
        merged_subs.append(previous_sub)
    return merged_subs

def parse_srt(srt_text):
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n{2}|\Z)', re.DOTALL)
    matches = pattern.findall(srt_text)
    subtitles = [{'index': int(m[0]), 'start': m[1], 'end': m[2], 'text': m[3].replace('\n', ' ')} for m in matches]
    return subtitles

def format_srt(subtitles):
    srt_text = ''
    for i, sub in enumerate(subtitles, start=1):
        srt_text += f"{i}\n{sub['start']} --> {sub['end']}\n{sub['text']}\n\n"
    return srt_text.strip()

def merge_and_format_srt(srt_text):
    subtitles = parse_srt(srt_text)
    merged_subs = merge_subtitles(subtitles)
    return format_srt(merged_subs)

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# Example usage
input_file = r"C:\Users\yongjie.yang\Desktop\The RISE of AI Agents (AI Agents Explained).en_dedup.srt"  # 输入文件路径
output_file = r"C:\Users\yongjie.yang\Desktop\The RISE of AI Agents (AI Agents Explained).en_dedup2.srt"  # 输出文件路径

srt_text = read_file(input_file)
merged_srt = merge_and_format_srt(srt_text)
write_file(output_file, merged_srt)

print(f"合并后的字幕文件已保存到 {output_file}")
