import re
import sys
import time
import tqdm
import pandas as pd
from pathlib import Path
from typing import Optional
from jsonargparse import CLI
import lightning as L
import torch

from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
# from generate.py script
from generate import generate


def main_generate_mqm(
    input_csv_file: str = "./mt-eval/mqm-classification/data/prompts_fewshot.csv",
    output_csv_file: str = "./mt-eval/mqm-classification/data/fewshot_outputs.csv",
    max_new_tokens: int = 5,
    top_k: int = 100,
    temperature: float = 0.5,
    checkpoint_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    quantize: Optional[str] = None,
):
    """
        quantize: Whether to quantize the model and using which method:
        ``"llm.int8"``: LLM.int8() mode,
        ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    ##### COPIED FROM main(...) IN generate.py #####
    if not checkpoint_path:
        checkpoint_path = Path(f"./checkpoints/lit-llama/7B/lit-llama.pth")
    if not tokenizer_path:
        tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
    assert checkpoint_path.is_file(), checkpoint_path
    assert tokenizer_path.is_file(), tokenizer_path

    fabric = L.Fabric(devices=1)
    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(checkpoint_path) as checkpoint:
        name = llama_model_lookup(checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=quantize
        ):
            model = LLaMA.from_name(name)

        model.load_state_dict(checkpoint)
    print(
        f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)
    L.seed_everything(1234)
    #################################################
    # load the csv with the prompts to generate on
    df = pd.read_csv(input_csv_file)
    # generate output for each prompt
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Classifying..."):
        prompt = row["prompt"]
        encoded_prompt = tokenizer.encode(
            prompt, bos=True, eos=False, device=fabric.device)
        y = generate(
            model=model,
            idx=encoded_prompt,
            max_new_tokens=max_new_tokens,
            # type: ignore[union-attr,arg-type]
            max_seq_length=model.config.block_size,
            temperature=temperature,
            top_k=top_k,
            eos_id=None
            # eos_id=13 #int(tokenizer.encode("\n")[-1]) # 13 is the token id for \n
        )
        # extract just the translation
        try:
            pattern_output = re.compile(r"Output: (.*)\n")
            match = pattern_output.search(tokenizer.decode(y))
            if match:
                df.loc[i, "me"] = match.group(1).strip()
        except:
            print(f"Error parsing output for row {i}: {row}")
    df.to_csv(output_csv_file, index=False)


if __name__ == "__main__":
    CLI(main_generate_mqm)
