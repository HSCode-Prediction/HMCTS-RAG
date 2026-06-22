import torch

from transformers import (
    GenerationConfig,
    #AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
)


def load_HF_model(ckpt) -> tuple:

    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Warning: tokenizer.pad_token is None. Setting it to tokenizer.eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    return tokenizer, model



def clean_model_output(output_text: str) -> str:
    return output_text.strip()



def generate_with_HF_model(
    tokenizer,
    model,
    input: str = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    num_beams: int = 1,
    num_return_sequences: int = 1,
    max_new_tokens: int = 128,
    **kwargs

):


    try:
        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to("cuda")
        do_sample_flag = True if num_return_sequences > 1 or temperature > 0.1 else False
        generation_config = GenerationConfig(
            do_sample=do_sample_flag,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
            )

        outputs = []
        for seq in generation_output:
            generated_tokens = seq[len(input_ids[0]):]
            decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            cleaned_text = clean_model_output(decoded_text)
            outputs.append(cleaned_text)
        return outputs[0] if num_return_sequences == 1 else outputs
    except Exception as e:

        import traceback
        traceback.print_exc()
        raise RuntimeError(f"HuggingFace generation failed: {str(e)}")

    return [] if num_return_sequences > 1 else ""