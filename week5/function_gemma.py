import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class FunctionGemmaCaller:
    def __init__(
        self,
        model_id: str = "google/functiongemma-270m-it",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,   # ✅ FIX
    ):
        self.device = device
        self.model_id = model_id

        print(f"Loading {model_id} on {device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use torch_dtype instead of dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )

        print(f"Model loaded successfully on {self.model.device}.")

    

    def run_inference(self, messages: list, tools: list[dict] = None, max_new_tokens: int = 512, skip_special_tokens: bool = False):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        attention_mask = torch.ones_like(inputs)    
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return self.tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=skip_special_tokens
        ).strip()
