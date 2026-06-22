import sys
sys.path.append(".")
from typing import List, Dict, Union

try:
    from vLLM_API import generate_with_vLLM_model
except ImportError:
    generate_with_vLLM_model = None

try:
    from OpenAI_API import generate_n_with_OpenAI_model, generate_prompts_with_OpenAI_model
except ImportError:
    generate_n_with_OpenAI_model, generate_prompts_with_OpenAI_model = None, None

try:
    from HuggingFace_API import generate_with_HF_model
except ImportError:
    generate_with_HF_model = None

class IO_System:

    def __init__(self, args, tokenizer, model) -> None:
        self.api = args.api
        self.model_ckpt = args.model_ckpt
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        
        self.tokenizer = tokenizer
        self.model = model
        
        self.call_counter = 0   # 统计 API 调用次数
        self.token_counter = 0  # 统计 总Token数 (Input + Output)

    def _count_tokens(self, text: Union[str, List[str]]) -> int:
        """辅助函数：计算文本的 Token 数量"""
        if not self.tokenizer:
            return 0
        
        count = 0
        text_list = [text] if isinstance(text, str) else text
        
        for t in text_list:
            try:
                # 优先使用 encode 计算
                if hasattr(self.tokenizer, 'encode'):
                    count += len(self.tokenizer.encode(t))
                else:
                    count += len(t) // 3
            except:
                pass
        return count

    def generate(self, model_input, max_tokens, num_return, stop_tokens=None) -> List[List[str]]:
        """
        统一生成接口
        """
        if stop_tokens is None:
            stop_tokens = []

        # 1. 增加调用计数
        self.call_counter += 1

        # 2. 统计输入 Token
        self.token_counter += self._count_tokens(model_input)

        # 确保输入是列表格式
        is_single_input = isinstance(model_input, str)
        if is_single_input:
            model_input = [model_input]

        all_outputs = []

        if self.api == "huggingface":
            if generate_with_HF_model is None:
                raise ImportError("Run_src Error: HuggingFace backend required but not installed.")
            
            for single_input in model_input:
                # 调用 HF 后端
                hf_response = generate_with_HF_model(
                    self.tokenizer, 
                    self.model, 
                    input=single_input, 
                    temperature=self.temperature, 
                    top_p=self.top_p, 
                    top_k=self.top_k, 
                    num_return_sequences=num_return, 
                    max_new_tokens=max_tokens
                )
                
                # 处理空返回
                if not hf_response:
                    all_outputs.append(["" for _ in range(num_return)])
                    continue
                
                if not isinstance(hf_response, list):
                    hf_response = [hf_response]
                
                # 补齐或截断
                if len(hf_response) < num_return:
                    hf_response.extend([hf_response[-1]] * (num_return - len(hf_response)))
                elif len(hf_response) > num_return:
                    hf_response = hf_response[:num_return]

                all_outputs.append(hf_response)

                # 3. 统计输出 Token (HF)
                self.token_counter += self._count_tokens(hf_response)

        elif self.api == "vllm":
            if generate_with_vLLM_model is None:
                raise ImportError("Run_src Error: vLLM backend required but not installed.")
            
            vllm_response = generate_with_vLLM_model(
                self.model, 
                input=model_input, 
                temperature=self.temperature, 
                top_p=self.top_p, 
                top_k=self.top_k, 
                n=num_return, 
                max_tokens=max_tokens, 
                stop=stop_tokens
            )
            
            batch_result = []
            for resp in vllm_response:
                single_prompt_outputs = []
                for o in resp.outputs:
                    text_out = o.text
                    single_prompt_outputs.append(text_out)
                    
                    # 3. 统计输出 Token (vLLM)
                    if hasattr(o, 'token_ids'):
                        self.token_counter += len(o.token_ids)
                    else:
                        self.token_counter += self._count_tokens(text_out)
                        
                batch_result.append(single_prompt_outputs)
            
            all_outputs = batch_result

        elif self.api == "gpt-4o" or self.api == "openai":
            if generate_n_with_OpenAI_model is None:
                raise ImportError("Run_src Error: OpenAI backend required but not installed.")
            
            for single_input in model_input:
                gpt_response = generate_n_with_OpenAI_model(
                    prompt=single_input,
                    n=num_return,
                    model_ckpt=self.model_ckpt,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    stop=stop_tokens
                )
                all_outputs.append(gpt_response)
                
                # 3. 统计输出 Token (OpenAI)
                self.token_counter += self._count_tokens(gpt_response)

        else:
            raise NotImplementedError(f"API '{self.api}' is not implemented.")
        
        # 返回结果
        if is_single_input:
            return all_outputs[0]
        else:
            return all_outputs