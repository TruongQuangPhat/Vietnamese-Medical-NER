import docx
import json
import re
import os
import glob
from underthesea import sent_tokenize
from tqdm import tqdm
import argparse


DICT_CONFIG = {
    "labels/list_herbs.txt": "HERB",
    "labels/list_diseases.txt": "DISEASE",
    "labels/list_symptoms.txt": "SYMPTOM",
    "labels/list_human_parts.txt": "HUMAN_PART",
    "labels/list_plant_parts.txt": "PLANT_PART"
}

ABBREVIATIONS = [
    "TS.", "ThS.", "PGS.", "GS.", "BS.", "DS.", "NXB.", "Dr.", "Prof.",
    "BN.", "ĐTĐ.", "THA.", "TP.", "Tp.",
    "v.v.",
    "L.", "Lour.", "Merr.", "Willd.", "Sw.", "Hook.", "Don.", "Wall.",
    "Decne.", "Benn.", "Roxb.", "Lamk.", "Gaud.", "Hemsl.", "Kurz.",
    "Fisch.", "tr.", "Tr.", "vol.", "Vol.", "p.", "pp."
]

SENTENCE_END_PUNCT = ['.', '!', '?']


def load_and_merge_dicts(config):
    """
    Load keyword dictionaries and merge them into a single list.
    Keywords are converted to lowercase and sorted by length
    in descending order to support longest-match-first strategy.
    """
    merged_list = []
    print("Loading dictionaries...")

    for filename, label_name in config.items():
        if not os.path.exists(filename):
            print(f"Warning: Dictionary file not found: {filename}")
            continue

        try:
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        merged_list.append((word.lower(), label_name))
        except Exception as e:
            print(f"Error reading dictionary {filename}: {e}")

    merged_list.sort(key=lambda x: len(x[0]), reverse=True)
    print(f"Loaded {len(merged_list)} keywords.")
    return merged_list


def clean_raw_line(text):
    """
    Clean a raw OCR line by removing common OCR artifacts
    and normalizing whitespace.
    """
    # Remove OCR bullets, pipes, or leading noise
    text = re.sub(r'^[\|\-•·]\s*', '', text)

    # Remove OCR trash blocks such as **** or ####
    text = re.sub(r'[\*#]{2,}', '', text)

    # Remove duplicated punctuation
    text = re.sub(r'([.,;:!?])\1+', r'\1', text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip()


def is_header(line):
    if line.isupper() and len(line) > 3:
        return True
    if re.match(r'^(CHƯƠNG|PHẦN|MỤC|BÀI)\s+\d+', line, re.IGNORECASE):
        return True
    if re.match(r'^[IVX]+\.', line):
        return True
    return False

def is_list_item_or_short(line):
    words = line.split()
    if re.match(r'^(\d+\.|\-|\•)', line):
        return True
    if len(words) <= 12 and not re.search(r'[.!?]$', line):
        return True
    return False

def ends_like_sentence(line):
    return bool(re.search(r'[.!?]["\']?$', line))

def likely_continuation(previous_line, current_line):
    if current_line[0].islower():
        return True
    if re.match(r'^(và|hoặc|nhưng|do đó|vì vậy|kết hợp)', current_line.lower()):
        return True
    if current_line[0] in ',;:':
        return True
    return False


def read_and_merge_docx(file_path):
    """
    Read a DOCX file and merge OCR-broken lines into
    coherent paragraph blocks.
    """
    if not os.path.exists(file_path):
        print(f"Error: Input file not found: {file_path}")
        return []

    try:
        doc = docx.Document(file_path)
    except Exception as e:
        print(f"Error opening DOCX file {file_path}: {e}")
        return []

    raw_lines = [
        clean_raw_line(p.text)
        for p in doc.paragraphs
        if p.text.strip()
    ]

    # Remove standalone page numbers
    raw_lines = [l for l in raw_lines if not re.fullmatch(r'\d+', l)]

    if not raw_lines:
        return []

    print(f"Processing {len(raw_lines)} OCR lines from {os.path.basename(file_path)}...")
    merged_paragraphs = []
    current_block = raw_lines[0]

    for next_line in tqdm(raw_lines[1:], desc="Merging OCR lines"):
        should_merge = False
        separator = " "

        if is_header(current_block) or is_header(next_line):
            should_merge = False

        elif is_list_item_or_short(current_block) and is_list_item_or_short(next_line):
            should_merge = True
            separator = ", "

        else:
            if not ends_like_sentence(current_block):
                should_merge = True
            elif likely_continuation(current_block, next_line):
                should_merge = True
            elif current_block.endswith(':'):
                should_merge = True

        if should_merge:
            current_block += separator + next_line
        else:
            merged_paragraphs.append(current_block)
            current_block = next_line

    merged_paragraphs.append(current_block)
    return merged_paragraphs


def protect_abbreviations(text):
    for abbr in ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", "@"))
    return text

def segment_into_sentences(paragraphs):
    final_sentences = []

    for para in tqdm(paragraphs, desc="Segmenting sentences"):
        if not para.strip():
            continue

        protected_text = protect_abbreviations(para)
        # Protect scientific author names
        protected_text = re.sub(
            r'\b([A-Z][a-z]{2,})\.',
            r'\1@',
            protected_text
        )

        try:
            sentences = sent_tokenize(protected_text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+', protected_text)

        for s in sentences:
            restored = s.replace("@", ".").strip()
            if len(restored) > 8 and any(c.isalpha() for c in restored):
                final_sentences.append(restored)

    return final_sentences

def find_dosage_matches(text):
    pattern = r'\b\d+([.,]\d+)?\s*(g|gam|gram|mg|ml|lít|l|viên|giọt|thang|chỉ|đồng cân|nắm|bó|củ|lát)\b'
    matches = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        matches.append([match.start(), match.end(), "DOSAGE"])
    return matches

def auto_annotate_multi(sentences, sorted_keywords):
    labeled_data = []
    print("Annotating sentences...")

    for sent in tqdm(sentences, desc="Annotating"):
        sent_lower = sent.lower()
        labels = []
        occupied_mask = [False] * len(sent)

        # 1. DOSAGE (High Priority)
        dosage_matches = find_dosage_matches(sent)
        for start, end, label in dosage_matches:
            labels.append([start, end, label])
            for k in range(start, end):
                occupied_mask[k] = True

        # 2. Dictionary Keywords
        for keyword, label_name in sorted_keywords:
            start_search = 0
            while True:
                idx = sent_lower.find(keyword, start_search)
                if idx == -1:
                    break

                end = idx + len(keyword)

                if (idx > 0 and sent_lower[idx - 1].isalnum()) or \
                   (end < len(sent_lower) and sent_lower[end].isalnum()):
                    start_search = idx + 1
                    continue

                if any(occupied_mask[idx:end]):
                    start_search = idx + 1
                    continue

                labels.append([idx, end, label_name])
                for i in range(idx, end):
                    occupied_mask[i] = True

                start_search = end

        if labels:
            labels.sort(key=lambda x: x[0])
            labeled_data.append({
                "text": sent,
                "label": labels
            })

    return labeled_data

def process_single_file(file_path, output_dir, keywords):
    """
    Process a single DOCX file and save the result to JSONL.
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"result_multi_label_{base_name}.jsonl"
    output_path = os.path.join(output_dir, output_filename)

    print(f"\n--- Processing: {base_name} ---")
    
    paragraphs = read_and_merge_docx(file_path)
    if not paragraphs:
        print(f"Skipping {base_name}: No text extracted.")
        return

    sentences = segment_into_sentences(paragraphs)
    print(f"Extracted {len(sentences)} sentences.")

    annotated_data = auto_annotate_multi(sentences, keywords)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in annotated_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Batch OCR-safe auto-labeling tool for Vietnamese Medical NER"
    )

    parser.add_argument(
        "-i", "--input_dir",
        default="input",
        help="Directory containing input DOCX files"
    )

    parser.add_argument(
        "-o", "--output_dir",
        default="output",
        help="Directory to save output JSONL files"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Starting Batch Processing ===")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")

    # Load keywords ONCE for all files
    keywords = load_and_merge_dicts(DICT_CONFIG)
    if not keywords:
        print("No keywords loaded. Exiting.")
        return

    # Find all .docx files in the input directory
    docx_pattern = os.path.join(args.input_dir, "*.docx")
    docx_files = glob.glob(docx_pattern)

    if not docx_files:
        print(f"No .docx files found in {args.input_dir}")
        return

    print(f"Found {len(docx_files)} files to process.")

    # Process each file
    for file_path in docx_files:
        try:
            process_single_file(file_path, args.output_dir, keywords)
        except Exception as e:
            print(f"ERROR processing file {file_path}: {e}")

    print("\n=== All tasks completed ===")

if __name__ == "__main__":
    main()

# python src/auto_labeling.py --input_dir "input" --output_dir "output"