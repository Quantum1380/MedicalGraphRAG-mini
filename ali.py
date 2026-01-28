import requests
import json
from typing import Optional, Dict, Any

class BaiLianAPI:
    """
    阿里百炼API调用封装类
    """
    # 百炼API基础地址
    BASE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def __init__(self, api_key: str):
        """
        初始化百炼API客户端
        :param api_key: 百炼API的API_KEY
        """
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    @staticmethod
    def _build_request_body(prompt: str, model: str = "qwen-turbo", **kwargs) -> Dict[str, Any]:
        """
        构建请求体（内部方法）
        :param prompt: 输入的提示词
        :param model: 使用的模型，默认qwen-turbo（通义千问轻量版）
        :param kwargs: 其他可选参数，如temperature、top_p等
        :return: 构建好的请求体字典
        """
        # 基础请求体
        body = {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "result_format": "message",
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.8)
            }
        }
        # 合并自定义参数
        if kwargs:
            body["parameters"].update(kwargs)
        return body

    def send_request(self, prompt: str, model: str = "qwen-turbo", timeout: int = 120, **kwargs) -> Dict[str, Any]:
        """
        发送请求到百炼API
        :param prompt: 用户输入的提示词
        :param model: 模型名称，如qwen-turbo/qwen-plus/qwen-max等
        :param timeout: 请求超时时间，默认120秒
        :param kwargs: 其他参数（temperature/top_p/max_tokens等）
        :return: API返回的响应数据（字典格式）
        """
        try:
            # 构建请求体
            request_body = self._build_request_body(prompt, model, **kwargs)

            # 发送POST请求
            response = requests.post(
                url=self.BASE_URL,
                headers=self.headers,
                data=json.dumps(request_body),
                timeout=timeout
            )

            # 检查响应状态码
            response.raise_for_status()

            # 返回解析后的JSON数据
            result = response.json()
            return result

        except requests.exceptions.Timeout:
            raise Exception(f"请求超时（{timeout}秒），请检查网络或重试")
        except requests.exceptions.ConnectionError:
            raise Exception("网络连接失败，请检查网络或API地址是否正确")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP请求错误: {e}, 响应内容: {response.text}")
        except Exception as e:
            raise Exception(f"请求失败: {str(e)}")

    def get_answer(self, prompt: str, model: str = "qwen-turbo", **kwargs) -> str:
        """
        获取AI的回答文本
        :param prompt: 用户输入的提示词
        :param model: 模型名称
        :param kwargs: 其他参数
        :return: AI返回的回答文本
        """
        result = self.send_request(prompt, model, **kwargs)
        try:
            answer = result["output"]["choices"][0]["message"]["content"]
            return answer
        except KeyError as e:
            raise Exception(f"解析回答失败，响应格式异常: {str(e)}, 原始响应: {result}")
