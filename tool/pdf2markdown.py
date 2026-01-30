import pdfplumber
import pandas as pd
import os


def pdf_to_markdown(pdf_path, output_md_path=None):
    """
    将 PDF 文件转换为 Markdown 格式

    Args:
        pdf_path (str): 输入的 PDF 文件路径
        output_md_path (str): 输出的 Markdown 文件路径，默认为原 PDF 同目录同名的 .md 文件

    Returns:
        str: 转换后的 Markdown 文本内容
    """
    # 设置默认输出路径
    if output_md_path is None:
        output_md_path = os.path.splitext(pdf_path)[0] + ".md"

    # 初始化 Markdown 内容
    md_content = []

    try:
        # 打开 PDF 文件
        with pdfplumber.open(pdf_path) as pdf:
            # 遍历每一页
            for page_num, page in enumerate(pdf.pages, 1):
                # 添加页码标题
                md_content.append(f"## 第 {page_num} 页\n")

                # 提取并处理文本
                text = page.extract_text()
                if text:
                    # 清理文本并转换为 markdown 格式
                    clean_text = text.strip()
                    # 简单的格式处理：换行符转换
                    clean_text = clean_text.replace('\n', '\n\n')
                    md_content.append(clean_text)
                    md_content.append("\n---\n")  # 分隔符

                # 提取并处理表格
                tables = page.extract_tables()
                if tables:
                    md_content.append("### 表格\n")
                    for table in tables:
                        # 将表格数据转换为 DataFrame
                        if table and len(table) > 0:
                            # 处理空单元格
                            table = [[cell.strip() if cell else "" for cell in row] for row in table]
                            df = pd.DataFrame(table[1:], columns=table[0] if len(table) > 1 else None)
                            # 转换为 markdown 表格
                            md_table = df.to_markdown(index=False)
                            md_content.append(md_table)
                            md_content.append("\n")

        # 将内容写入 Markdown 文件
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))

        print(f"转换完成！Markdown 文件已保存至：{output_md_path}")
        return '\n'.join(md_content)

    except FileNotFoundError:
        print(f"错误：找不到指定的 PDF 文件 - {pdf_path}")
        return ""
    except Exception as e:
        print(f"转换过程中出现错误：{str(e)}")
        return ""


# 示例用法
if __name__ == "__main__":
    # 替换为你的 PDF 文件路径
    input_pdf = "doc/Book_20250121.pdf"  # 输入PDF路径
    output_md = "Book_20250121.txt"  # 输出MD路径

    # 执行转换
    result = pdf_to_markdown(input_pdf, output_md)

    # 如果需要在控制台查看转换结果
    # print(result)