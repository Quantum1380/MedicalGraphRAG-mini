from ali import BaiLianAPI
from dotenv import load_dotenv
import os

# 加载.env文件
load_dotenv()

if __name__ == "__main__":
    API_KEY = os.getenv("BAILIAN_API_KEY")
    # 初始化阿里大模型
    llm = BaiLianAPI(api_key=API_KEY)

    try:
        prompt = "你是谁？"
        answer = llm.get_answer(
            prompt,
            model="qwen-turbo",
            temperature=0.5)
        print("\nAI回答：")
        print(answer)

    except Exception as e:
        print(f"调用失败: {e}")