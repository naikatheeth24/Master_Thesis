from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation import generate_output_from_input
import torch
import sys
import feedparser

def get_arxiv_paper_info(arxiv_id):
    # Base URL for arXiv API
    base_url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    
    # Parse the result
    parsed_data = feedparser.parse(base_url)
    
    # Extract title and summary
    if parsed_data.entries:
        entry = parsed_data.entries[0]
        title = entry.title.strip()
        summary = entry.summary.strip()
        return title, summary
    else:
        return None, None
      
if __name__ == "__main__":

    if len(sys.argv) <= 1:
        raise Exception('Must pass arxiv link id... python inference.py "2410.08196"')
      
    paper_link = sys.argv[1]

    model_id = "./models/peft_1/best" 
    # model_id = "meta-llama/Llama-3.2-1B-Instruct" # Base model
    batch_size = 16
    device = "cuda" # change to your device
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    
    model = AutoModelForCausalLM.from_pretrained(model_id,    
                                                torch_dtype=torch.bfloat16,
                                                device_map=device)


    title, abstract = get_arxiv_paper_info(arxiv_id="2410.08196")
    
    predictions = generate_output_from_input(model, tokenizer,
                                             title=title,
                                             abstract=abstract
                                             )
    print(f"\n\nTitle: {title}\nAbstract: {abstract[:500]}...\n\nPrediction:{predictions})")
