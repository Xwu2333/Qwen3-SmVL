"""
Vision-Language Model Inference Script for HuggingFace Models
This script loads a dataset from JSONL format and performs inference using a HuggingFace VL model.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from utils import load_model, load_processor
import time
import swanlab
from io import BytesIO
import csv
import datasets
# import datetime


class VLInferenceDataset:
    """Dataset class for loading JSONL inference data"""
    
    def __init__(self, jsonl_path: str, image_root: str = None):
        """
        Args:
            jsonl_path: Path to the JSONL file containing inference data
            image_root: Root directory for images. If None, uses directory of jsonl_path
        """
        self.jsonl_path = jsonl_path
        self.image_root = image_root or str(Path(jsonl_path).parent)
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file"""
        data = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data[:16]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset"""
        item = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.image_root, item['image_path'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
        
        return {
            'question_id': item['question_id'],
            'images': image,
            'texts': item['prompt'],
            'history': item.get('history', []),
            'image_path': image_path
        }


class VLModelInference:
    """Vision-Language Model Inference Handler"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        csv_file: str = "./temp/image_token_length.csv",
        trust_remote_code: bool = True,
        enable_csv: bool = True
    ):
        """
        Initialize the VL model for inference
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            device: Device to run inference on
            torch_dtype: Torch data type for model
            trust_remote_code: Whether to trust remote code (needed for some models)
        """
        self.device = device
        # self.model_name_or_path = model_name_or_path
        
        # print(f"Loading model from {model_name_or_path}...")
        # print(f"Using device: {device}")
        
        # Load tokenizer/processor
        print(f"正在加载训练后的模型: {checkpoint_path}")

        # 使用原始的模型构建方式
        self.model = load_model(self.device)
        self.processor = load_processor()
        self.csv_file = csv_file
        self.enable_csv = enable_csv
        self.call_count = 0
        
        # Initialize CSV file with headers if it doesn't exist or is empty
        if self.enable_csv:
            self._initialize_csv()
        self.model.model.connector.register_forward_hook(self.hook_function)
        
        # 加载训练后的权重
        if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
            print("正在加载safetensors权重...")
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
            self.model.load_state_dict(state_dict, strict=False)
            print("✅ 权重加载成功")
        elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
            print("正在加载pytorch权重...")
            state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location=device)
            self.model.load_state_dict(state_dict, strict=False)
            print("✅ 权重加载成功")
        else:
            print("⚠️  未找到权重文件，使用原始模型")
        
        self.model.eval()
    
    def generate(
        self,
        images,
        prompts,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Generate response(s) for image(s) and prompt(s)
        
        This function handles both single and batch inference automatically.
        
        Args:
            images: Single PIL Image or List of PIL Images
            prompts: Single text prompt (str) or List of text prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
        
        Returns:
            Single string if inputs are single, List of strings if inputs are batches
            
        Examples:
            # Single inference
            result = model.generate(image, "Describe this image")
            
            # Batch inference
            results = model.generate([img1, img2], ["Prompt 1", "Prompt 2"])
        """
        from transformers import set_seed

        # Check if inputs are single or batch
        is_single = not isinstance(images, list)
        
        # Convert to list format for unified processing
        if is_single:
            images = [images]
            prompts = [prompts]
        
        # Validate inputs
        if len(images) != len(prompts):
            raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")
        
        # Generate
        with torch.no_grad():
            try:
                # Prepare inputs - this will vary by model
                # Build messages for each prompt in the batch
                batch_texts = []
                for prompt in prompts:
                    messages = [
                        {
                            "role": "system",
                            "content": "使用中文回答所有问题。",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt},  # Add the actual prompt text here
                            ],
                        },
                    ]
                    
                    # Apply chat template for this prompt
                    text = self.processor.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    batch_texts.append(text)

                images_nested = [[img] for img in images]

                print("Batch image row shape before model processor: ", len(images_nested))
                total_img = 0
                for img in images_nested:
                    total_img += len(img)

                print("total number of images in batch:", total_img)
                
                # Now process the batch with all formatted texts
                inputs = self.processor(
                    text=batch_texts,  # List of formatted texts
                    images=images_nested,     # List of images
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                print("Batch image tensor shape after model processor:", inputs["pixel_values"].shape)
                print("Batch text tensor shape after model processor:", inputs["input_ids"].shape)

                for key in inputs:
                    if key == 'pixel_values' and inputs[key] is not None:
                        inputs[key] = inputs[key].to(torch.bfloat16)
                    elif key == 'input_ids' and inputs[key] is not None:
                        inputs[key] = inputs[key].to(self.device)
                    elif key == 'attention_mask' and inputs[key] is not None:
                        inputs[key] = inputs[key].to(self.device)

                set_seed(42)

                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    **kwargs
                )

                #only keep the model reponse part
                response_ids = []
                for output in outputs:
                    response_id = output[inputs["input_ids"].shape[1]:]
                    response_ids.append(response_id)
                    
                # Decode outputs
                generated_texts = self.processor.batch_decode(
                    response_ids,
                    skip_special_tokens=True
                )
                
                #Clean up outputs - remove thinking content
                cleaned_texts = []
                for text in generated_texts:
                    text = text.split("</think>")[-1]
                    cleaned_texts.append(text)
                
                # Return single string if input was single, else return list
                return cleaned_texts[0] if is_single else cleaned_texts
                
            except Exception as e:
                print(f"Error during generation: {e}")
                error_msg = f"[Generation Error: {str(e)}]"
                print(error_msg)
                # Return appropriate format based on input
                return error_msg if is_single else [error_msg for _ in range(len(images))]
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if needed."""
        if self.csv_file:
            file_exists = os.path.exists(self.csv_file)
            
            if not file_exists or os.path.getsize(self.csv_file) == 0:
                with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Write header - we'll use a flexible approach that adapts to dimensions
                    writer.writerow([
                        'call_number', 'output_type', 'shape_str',
                        'ndim', 'dim_0', 'dim_1', 'dim_2', 'dim_3', 'dim_4', 'dim_5'
                    ])
                print(f"✓ Created CSV file: {self.csv_file}")
    
    def _log_shape_to_csv(self, output):
        """
        Log output shape to CSV file.
        
        Args:
            output: The tensor or tuple output from the hooked layer
        """
        if not self.enable_csv:
            return
        
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.call_count += 1
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if isinstance(output, torch.Tensor):
                shape = list(output.shape)
                shape_str = str(shape)
                ndim = len(shape)
                
                # Pad dimensions to 6 (dim_0 through dim_5)
                dims = shape + [-1] * (6 - len(shape))
                
                writer.writerow([
                    self.call_count,
                    'Tensor',
                    shape_str,
                    ndim,
                    dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
                ])
                
            elif isinstance(output, tuple):
                # Log each element of the tuple
                for i, item in enumerate(output):
                    if isinstance(item, torch.Tensor):
                        shape = list(item.shape)
                        shape_str = f"tuple[{i}]: {shape}"
                        ndim = len(shape)
                        
                        dims = shape + [-1] * (6 - len(shape))
                        
                        writer.writerow([
                            self.call_count,
                            f'Tuple[{i}]',
                            shape_str,
                            ndim,
                            dims[0], dims[1], dims[2], dims[3], dims[4], dims[5]
                        ])
            print(f"✓ Logged to CSV: {self.csv_file}")
    def hook_function(self, module, input, output):
        """
        Hook function that will be called when the connector layer processes data.
        
        Args:
            module: The layer/module being hooked (connector in our case)
            input: Input tensor(s) to the layer
            output: Output tensor(s) from the layer
        """
        # Store the output for later analysis
        self.connector_output = output
        
        # Log to CSV
        self._log_shape_to_csv(output)
        
        # Print to console
        print(f"\n[Hook Triggered] Connector layer executed! (Call #{self.call_count})")
        print(f"Output type: {type(output)}")
        print(f"Hook module: {module}")
        
        if isinstance(output, torch.Tensor):
            print(f"Input: {input[0].shape}")
            print(f"Output shape: {output.shape}")
            print(f"Sequence length after connector: {output.shape[1]}")
        elif isinstance(output, tuple):
            print(f"Output is a tuple with {len(output)} elements")
            # for i, item in enumerate(output):
            #     if isinstance(item, torch.Tensor):
            #         print(f"  Element {i} shape: {item.shape}")


def run_inference(
    checkpoint_path: str,
    output_path: str,
    jsonl_path: Optional[str] = None,
    data: Optional[datasets.Dataset] = None,
    csv_path: Optional[str] = "",
    image_root: str = None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run inference on the dataset and save results
    
    Args:
        checkpoint_path: model local path
        jsonl_path: Path to input JSONL file
        output_path: Path to save output JSONL file
        image_root: Root directory for images
        batch_size: Batch size for inference (default: 8)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        device: Device to run inference on
    """
    if data:
        dataset = data
        print(f"Loaded {len(dataset)} samples")
    else:
        # Load dataset
        print(f"Loading dataset from {jsonl_path}...")
        dataset = VLInferenceDataset(jsonl_path, image_root)
        print(f"Loaded {len(dataset)} samples")
    
    # Initialize model
    model_inference = VLModelInference(
        checkpoint_path=checkpoint_path,
        csv_file = csv_path,
        device=device
    )
    
    # Run inference
    results = []
    print(f"\nRunning inference on {len(dataset)} samples with batch_size={batch_size} for model {checkpoint_path}...")
    
    # Process in batches
    num_batches = (len(dataset) + batch_size - 1) // batch_size

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    print("开始推理...")
    start_time = time.time()
    
    for batch_idx in tqdm(range(num_batches), desc="Inference"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(dataset))
        
        # Collect batch samples
        batch_samples = [dataset[i] for i in range(start_idx, end_idx)]
        
        # Prepare batch inputs
        print(batch_samples)
        # batch_images = [sample['images'][0] for sample in batch_samples]
        batch_images = [sample['images'] for sample in batch_samples]
        batch_prompts = [sample['texts'] for sample in batch_samples]
        if "question_id" in batch_samples[0].keys():
            batch_question_ids = [sample['question_id'] for sample in batch_samples]
        else:
            batch_question_ids = [i for i in range(len(batch_samples))]
        # Generate predictions for batch
        try:
            predictions = model_inference.generate(
                images=batch_images,
                prompts=batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
        except Exception as e:
            print(f"\nError processing batch {batch_idx}: {e}")
            print("Falling back to single-sample processing for this batch...")
            predictions = []
            for sample in batch_samples:
                try:
                    pred = model_inference.generate(
                        images=sample['images'],
                        prompts=sample['texts'],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    predictions.append(pred)
                except Exception as e2:
                    print(f"Error processing sample {sample['question_id']}: {e2}")
                    predictions.append(f"[Error: {str(e2)}]")
        
        # Store results
        for question_id, prediction in zip(batch_question_ids, predictions):
            result = {
                "question_id": question_id,
                "predict": prediction
            }
            results.append(result)
        
        # Save incrementally (every 100 samples) to avoid data loss
        if len(results) % 100 < batch_size and len(results) >= 100:
            save_results(results, output_path)
    
    end_time = time.time()
    print("推理完成...")
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_inf = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    inf_percentage = round(used_memory_for_inf / max_memory * 100, 3)
    print(f"{end_time - start_time:.4f} seconds used for inference.")
    print(
        f"{round((end_time - start_time)/60, 2)} minutes used for inference."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for inference = {used_memory_for_inf} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for inference % of max memory = {inf_percentage} %.")
    
    # Final save
    save_results(results, output_path)
    print(f"\nInference complete! Results saved to {output_path}")
    print(f"Total samples processed: {len(results)}")


def save_results(results: List[Dict[str, str]], output_path: str):
    """Save results to JSONL file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    """
    Main function - Configure your inference parameters here
    """

    swanlab.init(experiment_name = "freeze_vlm_llm_caudron_ZH_32K_1epoch_32batch_inference")
    # ===== CONFIGURATION =====
    # Model configuration
    MODEL_NAME_OR_PATH = "model/freeze_llm_vlm_cauldron_ZH_32K"  # Change this to your model
    # Examples of other models you can use:
    # - "Qwen/Qwen-VL-Chat"
    # - "OpenGVLab/InternVL-Chat-V1-5"
    # - "liuhaotian/llava-v1.5-7b"
    # - Or your local model path
    
    # Dataset configuration
    JSONL_PATH = "data/AlignMMBench/metadata.jsonl"  # Path to your input JSONL file
    IMAGE_ROOT = "data/AlignMMBench/"  # Root directory for images (None = same as JSONL directory)
    OUTPUT_PATH = "inference/inference_results_cauldron_ZH_32K_1epoch_32batch.jsonl"  # Output file path
    
    # Inference configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    BATCH_SIZE = 8  # Adjust based on your GPU memory (1, 2, 4, 8, 16, etc.)
    
    # ===== RUN INFERENCE =====
    run_inference(
        checkpoint_path=MODEL_NAME_OR_PATH,
        jsonl_path=JSONL_PATH,
        output_path=OUTPUT_PATH,
        image_root=IMAGE_ROOT,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        device=DEVICE
    )


if __name__ == "__main__":
    main()

