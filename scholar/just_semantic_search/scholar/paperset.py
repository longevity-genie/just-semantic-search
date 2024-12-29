from datasets import load_dataset

def download_aging_papers():
    """
    Downloads the aging papers dataset from Hugging Face.
    Returns the dataset object containing paragraphs about aging research.
    """
    dataset = load_dataset("longevity-genie/aging_papers_paragraphs")
    return dataset

if __name__ == "__main__":
    # Example usage
    dataset = download_aging_papers()
    print(f"Dataset loaded with {len(dataset['train'])} entries")
