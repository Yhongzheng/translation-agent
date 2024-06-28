import os
from typing import List
from typing import Union

import qianfan
import tiktoken
from dotenv import load_dotenv
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_env():
    # 获取当前脚本的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前脚本的上三级目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    # 构造 .env 文件的路径
    env_path = os.path.join(base_dir, '.env.qianfan')

    # 检查 .env 文件是否存在
    if os.path.exists(env_path):
        # 读取 .env 文件
        load_dotenv(dotenv_path=env_path)
        # print(f"Loaded .env file from: {env_path}")
    else:
        print(f".env file not found at: {env_path}")


# 调用函数加载 .env 文件
load_env()
# load_dotenv(dotenv_path='../../.env.qianfan')  # 读取本地 .env 文件

# 获取环境变量
qianfan_access_key = os.getenv("QIANFAN_ACCESS_KEY")
qianfan_secret_key = os.getenv("QIANFAN_SECRET_KEY")


# 设置环境变量
os.environ["QIANFAN_ACCESS_KEY"] = qianfan_access_key
os.environ["QIANFAN_SECRET_KEY"] = qianfan_secret_key

client = qianfan.ChatCompletion()

MAX_TOKENS_PER_CHUNK = (
    1000  # 如果文本超过这么多标记，我们将把它分成
)


# 离散的块，每次翻译一个块


def get_completion(
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        model: str = "ERNIE-Speed-128K",
        temperature: float = 0.1
) -> Union[str, dict]:
    """
    使用 OpenAI API 生成补全内容。

    参数:
        prompt (str): 用户的提示或查询。
        system_message (str, 可选): 设置助手上下文的系统消息。
            默认为 "You are a helpful assistant."。
        model (str, 可选): 用于生成补全内容的 OpenAI 模型名称。
            默认为 "gpt-4-turbo"。
        temperature (float, 可选): 控制生成文本随机性的采样温度。
            默认为 0.3。
        json_mode (bool, 可选): 是否以 JSON 格式返回响应。
            默认为 False。

    返回值:
        Union[str, dict]: 生成的补全内容。
            如果 json_mode 为 True，返回完整的 API 响应字典。
            如果 json_mode 为 False，返回生成的文本字符串。
    """

    resp = client.do(
        model=model,
        system=system_message,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ])
    return resp["body"]['result']


def one_chunk_initial_translation(
        source_lang: str, target_lang: str, source_text: str
) -> str:
    """
    Translate the entire text as one chunk using an LLM.

    Args:
        source_lang (str): The source language of the text.
        target_lang (str): The target language for translation.
        source_text (str): The text to be translated.

    Returns:
        str: The translated text.

    使用大型语言模型将整段文本进行翻译。

    参数:
        source_lang (str): 源语言的代码。
        target_lang (str): 目标语言的代码。
        source_text (str): 要翻译的文本。

    返回值:
        str: 翻译后的文本。
    """

    system_message = f"您是一名翻译专家，专门从事从 {source_lang} 到 {target_lang} 的翻译。"

    translation_prompt = f"""这是一个从 {source_lang} 到 {target_lang} 的翻译请求，请提供该文本的 {target_lang} 译文。
    请不要提供任何解释或除翻译外的文本。
{source_lang}: {source_text}

{target_lang}:"""

    prompt = translation_prompt.format(source_text=source_text)

    translation = get_completion(prompt, system_message=system_message)

    return translation


def one_chunk_reflect_on_translation(
        source_lang: str,
        target_lang: str,
        source_text: str,
        translation_1: str,
        country: str = "",
) -> str:
    """
    使用大型语言模型对翻译进行反思，将整段文本作为一个整体处理。

    参数:
        source_lang (str): 源文本的语言代码。
        target_lang (str): 翻译文本的目标语言代码。
        source_text (str): 源文本的原始内容。
        translation_1 (str): 源文本的初次翻译。
        country (str): 目标语言对应的国家。

    返回值:
        str: 大型语言模型对翻译的反思，提供建设性的批评和改进建议。
    """

    system_message = f"""您是一名翻译专家，专门从事从 {source_lang} 到 {target_lang} 的翻译。
    您将收到一个源文本及其翻译，您的目标是改进这段翻译。"""

    if country != "":
        reflection_prompt = f"""您的任务是仔细阅读从 {source_lang} 到 {target_lang} 的源文本和翻译，然后提供建设性的批评和有帮助的改进建议。\
    翻译的最终风格和语气应与在 {country} 口语中使用的 {target_lang} 风格相匹配。

    源文本和初次翻译如下，以 XML 标签 <SOURCE_TEXT></SOURCE_TEXT> 和 <TRANSLATION></TRANSLATION> 分隔：

    <SOURCE_TEXT>
    {source_text}
    </SOURCE_TEXT>

    <TRANSLATION>
    {translation_1}
    </TRANSLATION>

    在编写建议时，请注意是否有改进翻译的方法：
    (i) 准确性（通过纠正添加、误译、遗漏或未翻译的错误），
    (ii) 流畅性（通过应用 {target_lang} 语法、拼写和标点符号规则，确保没有不必要的重复），
    (iii) 风格（确保翻译反映源文本的风格，并考虑任何文化背景），
    (iv) 术语（确保术语使用一致并反映源文本领域；仅确保使用 {target_lang} 中的等效习语）。

    写出一份具体、有帮助和建设性的改进翻译建议的清单。
    每条建议应针对翻译的一个具体部分。
    只输出建议，不要输出其他内容。"""

    else:
        reflection_prompt = f"""您的任务是仔细阅读从 {source_lang} 到 {target_lang} 的源文本和翻译，然后提供建设性的批评和有帮助的改进建议。

    源文本和初次翻译如下，以 XML 标签 <SOURCE_TEXT></SOURCE_TEXT> 和 <TRANSLATION></TRANSLATION> 分隔：

    <SOURCE_TEXT>
    {source_text}
    </SOURCE_TEXT>

    <TRANSLATION>
    {translation_1}
    </TRANSLATION>

    在编写建议时，请注意是否有改进翻译的方法：
    (i) 准确性（通过纠正添加、误译、遗漏或未翻译的错误），
    (ii) 流畅性（通过应用 {target_lang} 语法、拼写和标点符号规则，确保没有不必要的重复），
    (iii) 风格（确保翻译反映源文本的风格，并考虑任何文化背景），
    (iv) 术语（确保术语使用一致并反映源文本领域；仅确保使用 {target_lang} 中的等效习语）。

    写出一份具体、有帮助和建设性的改进翻译建议的清单。
    每条建议应针对翻译的一个具体部分。
    只输出建议，不要输出其他内容。"""

    prompt = reflection_prompt.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        translation_1=translation_1,
    )
    reflection = get_completion(prompt, system_message=system_message)
    return reflection


def one_chunk_improve_translation(
        source_lang: str,
        target_lang: str,
        source_text: str,
        translation_1: str,
        reflection: str,
) -> str:
    """
    根据反思改进翻译，将整段文本作为一个整体处理。

    参数:
        source_lang (str): 源文本的语言代码。
        target_lang (str): 翻译文本的目标语言代码。
        source_text (str): 源文本的原始内容。
        translation_1 (str): 源文本的初次翻译。
        reflection (str): 改进翻译的专家建议和建设性批评。

    返回值:
        str: 基于专家建议改进后的翻译。
    """

    system_message = f"您是一名翻译编辑专家，专门从事从 {source_lang} 到 {target_lang} 的翻译编辑。"

    prompt = f"""您的任务是仔细阅读并编辑从 {source_lang} 到 {target_lang} 的翻译，参考专家建议和建设性批评。

    源文本、初次翻译和专家建议如下，以 XML 标签 <SOURCE_TEXT></SOURCE_TEXT>、<TRANSLATION></TRANSLATION> 和 <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> 分隔：

    <SOURCE_TEXT>
    {source_text}
    </SOURCE_TEXT>

    <TRANSLATION>
    {translation_1}
    </TRANSLATION>

    <EXPERT_SUGGESTIONS>
    {reflection}
    </EXPERT_SUGGESTIONS>

    请在编辑翻译时参考专家建议。编辑翻译时，请确保：

    (i) 准确性（通过纠正添加、误译、遗漏或未翻译的错误），
    (ii) 流畅性（通过应用 {target_lang} 语法、拼写和标点符号规则，确保没有不必要的重复），
    (iii) 风格（确保翻译反映源文本的风格），
    (iv) 术语（上下文不合适、不一致的使用），
    (v) 其他错误。

    只输出新的翻译，不要输出其他内容。"""

    translation_2 = get_completion(prompt, system_message)

    return translation_2


def one_chunk_translate_text(
        source_lang: str, target_lang: str, source_text: str, country: str = ""
) -> str:
    """
    将整段文本从源语言翻译为目标语言。

    此函数执行两步翻译过程：
    1. 获取源文本的初次翻译。
    2. 反思初次翻译并生成改进后的翻译。

    参数:
        source_lang (str): 源文本的语言代码。
        target_lang (str): 翻译文本的目标语言代码。
        source_text (str): 要翻译的文本。
        country (str): 指定目标语言的国家。

    返回值:
        str: 改进后的源文本翻译。
    """

    translation_1 = one_chunk_initial_translation(
        source_lang, target_lang, source_text
    )

    reflection = one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, translation_1, country
    )
    translation_2 = one_chunk_improve_translation(
        source_lang, target_lang, source_text, translation_1, reflection
    )

    return translation_2


def num_tokens_in_string(
        input_str: str, encoding_name: str = "cl100k_base"
) -> int:
    """
    使用指定的编码计算给定字符串中的标记数。

    参数:
        str (str): 要进行标记化的输入字符串。
        encoding_name (str, 可选): 要使用的编码名称。默认为 "cl100k_base"，这是最常用的编码器（由 GPT-4 使用）。

    返回值:
        int: 输入字符串中的标记数。

    示例:
        >>> text = "Hello, how are you?"
        >>> num_tokens = num_tokens_in_string(text)
        >>> print(num_tokens)
        5
    """

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens


def multichunk_initial_translation(
        source_lang: str, target_lang: str, source_text_chunks: List[str]
) -> List[str]:
    """
    将多段文本从源语言翻译为目标语言。

    参数:
        source_lang (str): 文本的源语言。
        target_lang (str): 翻译的目标语言。
        source_text_chunks (List[str]): 需要翻译的文本段列表。

    返回:
        List[str]: 翻译后的文本段列表。
    """

    system_message = f"你是一位语言学专家，专门从事将 {source_lang} 翻译成 {target_lang} 的工作。"

    translation_prompt = """你的任务是将部分文本从 {source_lang} 专业翻译为 {target_lang}。

源文本如下，由 XML 标签 <SOURCE_TEXT> 和 </SOURCE_TEXT> 分隔。仅翻译源文本中由 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 分隔的部分。你可以使用其余的源文本作为上下文，但不要翻译任何其他文本。仅输出所需翻译部分的翻译结果。

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

再次强调，你只需翻译此部分文本，再次展示在 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 之间的内容：
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

仅输出你被要求翻译的部分的翻译结果，除此之外不输出任何内容。
"""

    translation_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        tagged_text = (
                "".join(source_text_chunks[0:i])
                + "<TRANSLATE_THIS>"
                + source_text_chunks[i]
                + "</TRANSLATE_THIS>"
                + "".join(source_text_chunks[i + 1:])
        )

        prompt = translation_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
        )

        translation = get_completion(prompt, system_message=system_message)
        translation_chunks.append(translation)

    return translation_chunks


def multichunk_reflect_on_translation(
        source_lang: str,
        target_lang: str,
        source_text_chunks: List[str],
        translation_1_chunks: List[str],
        country: str = "",
) -> List[str]:
    """
    提供建设性的批评和改进部分翻译的建议。

    参数:
        source_lang (str): 文本的源语言。
        target_lang (str): 翻译的目标语言。
        source_text_chunks (List[str]): 被分成块的源文本。
        translation_1_chunks (List[str]): 对应源文本块的翻译块。
        country (str): 指定目标语言的国家。

    返回:
        List[str]: 反思的列表，包含对每个翻译块的改进建议。
    """

    system_message = f"你是一位语言学专家，专门从事将 {source_lang} 翻译成 {target_lang} 的工作。你将获得一段源文本及其翻译，你的目标是改进该翻译。"

    if country != "":
        reflection_prompt = """你的任务是仔细阅读一段从 {source_lang} 翻译成 {target_lang} 的源文本及其部分翻译，然后提供建设性的批评和有用的建议，以改进翻译。最终的翻译风格和语气应与在 {country} 日常口语中的 {target_lang} 风格相匹配。

源文本如下，由 XML 标签 <SOURCE_TEXT> 和 </SOURCE_TEXT> 分隔，已翻译的部分在源文本中由 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 分隔。你可以使用其余的源文本作为评价翻译部分的上下文。

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

再次强调，仅翻译了部分文本，再次展示在 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 之间的内容：
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

以下是标明的部分的翻译，分隔在 <TRANSLATION> 和 </TRANSLATION> 之间：
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

在撰写建议时，请注意是否有改进翻译的方式：
(i) 准确性（通过纠正添加、误译、遗漏或未翻译的文本错误），
(ii) 流利度（通过应用 {target_lang} 语法、拼写和标点规则，并确保没有不必要的重复），
(iii) 风格（通过确保翻译反映源文本的风格，并考虑任何文化背景），
(iv) 术语（通过确保术语使用一致并反映源文本领域；并确保仅使用相当于 {target_lang} 的等效成语）。

写下具体、有帮助和建设性的改进翻译的建议列表。每条建议应针对翻译的一个具体部分。仅输出建议内容，除此之外不输出任何内容。"""

    else:
        reflection_prompt = """你的任务是仔细阅读一段从 {source_lang} 翻译成 {target_lang} 的源文本及其部分翻译，然后提供建设性的批评和有用的建议，以改进翻译。

源文本如下，由 XML 标签 <SOURCE_TEXT> 和 </SOURCE_TEXT> 分隔，已翻译的部分在源文本中由 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 分隔。你可以使用其余的源文本作为评价翻译部分的上下文。

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

再次强调，仅翻译了部分文本，再次展示在 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 之间的内容：
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

以下是标明的部分的翻译，分隔在 <TRANSLATION> 和 </TRANSLATION> 之间：
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

在撰写建议时，请注意是否有改进翻译的方式：
(i) 准确性（通过纠正添加、误译、遗漏或未翻译的文本错误），
(ii) 流利度（通过应用 {target_lang} 语法、拼写和标点规则，并确保没有不必要的重复），
(iii) 风格（通过确保翻译反映源文本的风格，并考虑任何文化背景），
(iv) 术语（通过确保术语使用一致并反映源文本领域；并确保仅使用相当于 {target_lang} 的等效成语）。

写下具体、有帮助和建设性的改进翻译的建议列表。每条建议应针对翻译的一个具体部分。仅输出建议内容，除此之外不输出任何内容。"""

    reflection_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        tagged_text = (
                "".join(source_text_chunks[0:i])
                + "<TRANSLATE_THIS>"
                + source_text_chunks[i]
                + "</TRANSLATE_THIS>"
                + "".join(source_text_chunks[i + 1:])
        )
        if country != "":
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                translation_1_chunk=translation_1_chunks[i],
                country=country,
            )
        else:
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                translation_1_chunk=translation_1_chunks[i],
            )

        reflection = get_completion(prompt, system_message=system_message)
        reflection_chunks.append(reflection)

    return reflection_chunks


def multichunk_improve_translation(
        source_lang: str,
        target_lang: str,
        source_text_chunks: List[str],
        translation_1_chunks: List[str],
        reflection_chunks: List[str],
) -> List[str]:
    """
    改进从源语言到目标语言的文本翻译，参考专家建议。
    参数:
        source_lang (str): 文本的源语言。
        target_lang (str): 翻译的目标语言。
        source_text_chunks (List[str]): 分块后的源文本。
        translation_1_chunks (List[str]): 每个块的初始翻译。
        reflection_chunks (List[str]): 专家对每个翻译块的改进建议。
    返回:
        List[str]: 改进后的每个翻译块。
    """

    system_message = f"您是一位语言学专家，专门从事从 {source_lang} 到 {target_lang} 的翻译编辑工作。"

    improvement_prompt = """您的任务是仔细阅读并改进从 {source_lang} 到 {target_lang} 的翻译，考虑一组专家建议和建设性批评。以下提供了源文本、初始翻译和专家建议。

源文本如下，由XML标签 <SOURCE_TEXT> 和 </SOURCE_TEXT> 分隔，要翻译的部分在源文本中的 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 之间。您可以将源文本的其余部分用作上下文，但只需翻译 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 之间的部分。

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

再次重申，只翻译部分文本，再次显示在 <TRANSLATE_THIS> 和 </TRANSLATE_THIS> 之间：
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

以下是该部分文本的翻译，由 <TRANSLATION> 和 </TRANSLATION> 分隔：
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

以下是专家对该部分文本的翻译建议，由 <EXPERT_SUGGESTIONS> 和 </EXPERT_SUGGESTIONS> 分隔：
<EXPERT_SUGGESTIONS>
{reflection_chunk}
</EXPERT_SUGGESTIONS>

考虑专家建议，改写翻译以改进其

(i) 准确性（通过纠正添加、误译、省略或未翻译的错误），
(ii) 流畅性（应用 {target_lang} 语法、拼写和标点规则，并确保没有不必要的重复），
(iii) 风格（确保翻译反映源文本的风格），
(iv) 术语（不适合上下文、不一致的使用），或
(v) 其他错误。

仅输出修订后的翻译，不要包括其他内容。"""

    translation_2_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        tagged_text = (
                "".join(source_text_chunks[0:i])
                + "<TRANSLATE_THIS>"
                + source_text_chunks[i]
                + "</TRANSLATE_THIS>"
                + "".join(source_text_chunks[i + 1:])
        )

        prompt = improvement_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
            translation_1_chunk=translation_1_chunks[i],
            reflection_chunk=reflection_chunks[i],
        )

        translation_2 = get_completion(prompt, system_message=system_message)
        translation_2_chunks.append(translation_2)

    return translation_2_chunks


def multichunk_translation(
        source_lang, target_lang, source_text_chunks, country: str = ""
):
    """
    改进基于初始翻译和反思的多个文本片段翻译。

    参数：
    source_lang (str)：文本片段的源语言。
    target_lang (str)：翻译的目标语言。
    source_text_chunks (List[str])：需要翻译的源文本片段列表。
    translation_1_chunks (List[str])：每个源文本片段的初始翻译列表。
    reflection_chunks (List[str])：对初始翻译的反思列表。
    country (str)：目标语言指定的国家。

    返回：
    List[str]：每个源文本片段的改进翻译列表。
    """

    translation_1_chunks = multichunk_initial_translation(
        source_lang, target_lang, source_text_chunks
    )

    reflection_chunks = multichunk_reflect_on_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        country,
    )

    translation_2_chunks = multichunk_improve_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        reflection_chunks,
    )

    return translation_2_chunks


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    """
    计算块大小基于令牌数和令牌限制。

    参数:
    token_count (int): 总令牌数。
    token_limit (int): 每块允许的最大令牌数。

    返回:
        int: 计算出的块大小。

    描述:
        此函数根据给定的令牌数和令牌限制计算块大小。
        如果令牌数小于或等于令牌限制，函数将返回令牌数作为块大小。
        否则，它将计算需要多少块来容纳所有令牌在令牌限制内。
        块大小通过将令牌限制除以块数来确定。
        如果在将令牌数除以令牌限制后有剩余令牌，
        块大小将通过将剩余令牌除以块数来进行调整。

    示例:
        >>> calculate_chunk_size(1000, 500)
        500
        >>> calculate_chunk_size(1530, 500)
        389
        >>> calculate_chunk_size(2242, 500)
        496
    """

    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size


def translate(
        source_lang,
        target_lang,
        source_text,
        country,
        max_tokens=MAX_TOKENS_PER_CHUNK,
):
    """将 source_text 从 source_lang 翻译成 target_lang 。"""

    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as single chunk")

        final_translation = one_chunk_translate_text(
            source_lang, target_lang, source_text, country
        )

        return final_translation

    else:
        ic("Translating text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        translation_2_chunks = multichunk_translation(
            source_lang, target_lang, source_text_chunks, country
        )

        return "".join(translation_2_chunks)
