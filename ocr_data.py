import os
import re
import cv2
import torch
import numpy as np
import gc
import argparse
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from tqdm import tqdm

# --- MODEL DEPENDENCIES ---
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from easyocr import Reader

# --- CONFIGURATION SETTINGS ---
Image.MAX_IMAGE_PIXELS = None  # Prevent DecompressionBombError

class Config:
    DPI_RESOLUTION = 300
    TOLERANCE_Y = 30
    MARGIN_HEADER = 0.08
    MARGIN_FOOTER = 0.08
    BATCH_SIZE = 10
    
    # Regex for Level 2 headers (e.g., 2.1, 6.2.1)
    REGEX_HEADER_L2 = re.compile(r"^\s*(\d+(?:\.\d+)+)\.?\s+(?P<content>\S.*)$")

# --- INITIALIZE COMPUTATION DEVICE ---
COMPUTE_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()

print(f"System initialized on device: {COMPUTE_DEVICE.upper()}")

# --- LOAD MODELS ---
print("Initializing VietOCR engine...")
voc_cfg = Cfg.load_config_from_name("vgg_transformer")
voc_cfg["cnn"]["pretrained"] = False
voc_cfg["device"] = COMPUTE_DEVICE
voc_cfg["predictor"]["beamsearch"] = False
text_recognizer = Predictor(voc_cfg)

print("Initializing EasyOCR engine...")
text_detector = Reader(["vi"], gpu=HAS_CUDA, quantize=False)


# ==============================================================================
# CORE LOGIC: IMAGE EXTRACTION
# ==============================================================================
def extract_visual_elements(source_image: Image.Image, output_folder: str, page_index: int) -> List[Dict]:
    """
    Detects non-text elements (images/charts) using morphological operations.
    FIXED: Renamed argument 'page_idx' to 'page_index' to match internal usage.
    """
    image_arr = np.asarray(source_image)
    img_h, img_w = image_arr.shape[:2]
    
    # Preprocessing
    grayscale = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological dilation to merge features
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated_map = cv2.dilate(bin_img, morph_kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_items = []
    element_counter = 1
    total_area = img_w * img_h

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        curr_area = w * h

        # 1. Filter by dimensions relative to page size
        if curr_area < total_area * 0.015 or curr_area > total_area * 0.75:
            continue
        
        # 2. Filter by aspect ratio (avoid full columns or thin lines)
        if w / img_w > 0.9 or h / img_h > 0.9:
            continue
        
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.3 or aspect_ratio > 6:
            continue

        # 3. Filter by pixel density (text blocks usually have low density)
        region_of_interest = bin_img[y:y+h, x:x+w]
        pixel_density = cv2.countNonZero(region_of_interest) / region_of_interest.size
        
        if pixel_density > 0.12:
            continue 

        # Export valid image region
        cropped_region = image_arr[y:y+h, x:x+w]
        # SỬA LỖI Ở ĐÂY: page_index đã được định nghĩa đúng ở tham số hàm
        filename = f"p{page_index + 1}_img{element_counter}.jpg"
        full_save_path = os.path.join(output_folder, filename)
        
        Image.fromarray(cropped_region).save(full_save_path, quality=95)

        extracted_items.append({
            "type": "image",
            "path": full_save_path,
            "filename": filename,
            "y_center": (y + h/2) / img_h
        })
        element_counter += 1

    return extracted_items


# ==============================================================================
# CORE LOGIC: HEADER VALIDATION
# ==============================================================================
def validate_l2_header(text_content: str) -> bool:
    """
    Validates if a line is a Level 2 header (e.g. 2.1) using regex and heuristics.
    """
    if not text_content or len(text_content.strip()) < 4:
        return False
    
    clean_str = text_content.strip()
    
    # 1. Check against blacklist words
    forbidden_starts = ["Hình", "Hinh", "Bảng", "Bang", "Sơ đồ", "Biểu đồ", "Lược đồ", "Box"]
    lower_str = clean_str.lower()
    for word in forbidden_starts:
        if lower_str.startswith(word.lower()):
            return False

    # 2. Check regex pattern match
    match_obj = Config.REGEX_HEADER_L2.match(clean_str)
    if not match_obj:
        return False
    
    # 3. Verify numeric structure (Level 2 requires exactly 2 parts, e.g., 2.1)
    # 2.1.1 is Level 3 -> False
    number_segment = match_obj.group(1)
    numeric_parts = [n for n in number_segment.strip(".").split(".") if n.isdigit()]
    
    if len(numeric_parts) != 2:
        return False

    # 4. Content analysis to avoid measurement units
    content_segment = match_obj.group("content")
    unit_regex = r"^(g|gr|mg|kg|ml|lít|tốn|thốn|cm|mm|m|độ|%|triệu|tỷ|nghìn)(\s|$|\W)"
    
    if re.search(unit_regex, content_segment, re.IGNORECASE):
        return False
        
    # Ensure there is at least one alphabetic character
    if not re.search(r"[a-zA-Z]", content_segment):
        return False

    return True


# ==============================================================================
# CORE LOGIC: TEXT RECOGNITION
# ==============================================================================
def process_page_ocr(img_array: np.ndarray) -> List[Dict]:
    img_h, img_w = img_array.shape[:2]
    
    # Step 1: Detect text boxes
    try:
        detection_results = text_detector.readtext(img_array, detail=1, width_ths=0.7)
    except Exception:
        return []

    processed_text_blocks = []
    safe_y_min = img_h * Config.MARGIN_HEADER
    safe_y_max = img_h * (1 - Config.MARGIN_FOOTER)

    # Step 2: Filter and Recognize
    for bbox, _, confidence in detection_results:
        if confidence < 0.25:
            continue
        
        y_coords = [pt[1] for pt in bbox]
        y_center = sum(y_coords) / 4
        
        # Skip headers and footers
        if y_center < safe_y_min or y_center > safe_y_max:
            continue

        x_coords = [pt[0] for pt in bbox]
        x_min, y_min = int(min(x_coords)), int(min(y_coords))
        x_max, y_max = int(max(x_coords)), int(max(y_coords))

        # Crop text region
        cropped_text = img_array[
            max(0, y_min - 4) : min(img_h, y_max + 4),
            max(0, x_min - 4) : min(img_w, x_max + 4)
        ]
        
        if cropped_text.size == 0:
            continue

        try:
            pil_crop = Image.fromarray(cropped_text)
            recognized_str = text_recognizer.predict(pil_crop)
            
            if len(recognized_str.strip()) > 1:
                processed_text_blocks.append({
                    "text": recognized_str.strip(),
                    "y_center": y_center / img_h,
                    "x_min": min(x_coords)
                })
        except Exception:
            continue

    if not processed_text_blocks:
        return []
    
    # Step 3: Line Merging Algorithm
    processed_text_blocks.sort(key=lambda item: item["y_center"])
    
    merged_lines = []
    current_line_group = [processed_text_blocks[0]]
    tolerance_ratio = Config.TOLERANCE_Y / img_h

    for i in range(1, len(processed_text_blocks)):
        current_block = processed_text_blocks[i]
        prev_block = current_line_group[-1]
        
        # Check vertical distance
        if abs(current_block["y_center"] - prev_block["y_center"]) <= tolerance_ratio:
            current_line_group.append(current_block)
        else:
            merged_lines.append(current_line_group)
            current_line_group = [current_block]
    merged_lines.append(current_line_group)

    # Step 4: Construct Final Line Strings
    final_output = []
    for line_group in merged_lines:
        # Sort words left-to-right
        line_group.sort(key=lambda item: item["x_min"])
        joined_text = " ".join([block["text"] for block in line_group])
        avg_y = sum(block["y_center"] for block in line_group) / len(line_group)
        
        final_output.append({
            "type": "text",
            "content": joined_text,
            "y_center": avg_y
        })
        
    return final_output


# ==============================================================================
# OUTPUT GENERATION
# ==============================================================================
def generate_word_document(content_list: List[Dict], save_path: str):
    doc_obj = Document()
    default_style = doc_obj.styles["Normal"]
    default_style.font.name = "Times New Roman"
    default_style.font.size = Pt(13)
    
    section_buffer = []
    
    for item in content_list:
        if item["type"] == "text":
            text_str = item["content"]
            
            # Identify Section Break
            if validate_l2_header(text_str):
                print(f"[NEW SECTION DETECTED] {text_str[:50]}...")
                
                # Flush existing buffer
                if section_buffer:
                    for elem in section_buffer:
                        if isinstance(elem, str): 
                            doc_obj.add_paragraph(elem)
                        else: 
                            # Image Handling
                            try:
                                doc_obj.add_picture(elem["path"], width=Inches(5.0))
                                caption = doc_obj.add_paragraph()
                                runner = caption.add_run(f"[IMG-ID: {elem['filename']}]")
                                runner.font.color.rgb = RGBColor(0, 0, 255)
                                runner.font.size = Pt(10)
                                runner.italic = True
                                caption.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                            except Exception:
                                pass
                    
                    # Insert Separation Tag
                    sep_para = doc_obj.add_paragraph()
                    sep_run = sep_para.add_run("</break>")
                    sep_run.font.color.rgb = RGBColor(255, 0, 0)
                    sep_run.font.bold = True
                    sep_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                    
                    section_buffer = [] # Reset
                
                section_buffer.append(text_str)
            else:
                section_buffer.append(text_str)
        
        elif item["type"] == "image":
            section_buffer.append(item)
            
    # Write remaining buffer content
    if section_buffer:
        for elem in section_buffer:
            if isinstance(elem, str): 
                doc_obj.add_paragraph(elem)
            else:
                try:
                    doc_obj.add_picture(elem["path"], width=Inches(5.0))
                    caption = doc_obj.add_paragraph()
                    runner = caption.add_run(f"[IMG-ID: {elem['filename']}]")
                    runner.font.color.rgb = RGBColor(0, 0, 255)
                    runner.font.size = Pt(10)
                    runner.italic = True
                    caption.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                except Exception:
                    pass
    
    doc_obj.save(save_path)


# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================
def orchestrate_pipeline(source_path: str, start_page: int = 1, end_page: Optional[int] = None):
    if not os.path.exists(source_path):
        print(f"Error: Input file does not exist: {source_path}")
        return

    # Setup directories
    f_name = os.path.basename(source_path)
    f_base = os.path.splitext(f_name)[0]
    
    root_out = "output_results"
    specific_out_dir = os.path.join(root_out, f"{f_base}_result")
    images_out_dir = os.path.join(specific_out_dir, "images")
    
    os.makedirs(images_out_dir, exist_ok=True)

    print(f"Processing target: {f_name}")
    
    # Determine page count
    try:
        pdf_meta = pdfinfo_from_path(source_path)
        total_pages = pdf_meta["Pages"]
        print(f"Total pages detected: {total_pages}")
    except Exception:
        total_pages = 1000
        print("Warning: Unable to read PDF metadata. Defaulting to safe limit.")

    target_end = min(end_page, total_pages) if end_page else total_pages
    print(f"Processing Range: {start_page} -> {target_end}")

    aggregated_data = []
    
    # Batch Processing Loop
    for batch_start in range(start_page, target_end + 1, Config.BATCH_SIZE):
        batch_end = min(batch_start + Config.BATCH_SIZE - 1, target_end)
        
        print(f"\nProcessing Batch: {batch_start} to {batch_end}")
        
        try:
            pdf_pages = convert_from_path(source_path, dpi=Config.DPI_RESOLUTION, first_page=batch_start, last_page=batch_end)
        except Exception as err:
            print(f"Error converting PDF batch: {err}")
            continue

        for i, page_img in enumerate(pdf_pages):
            real_page_idx = batch_start + i - 1
            
            # Convert to numpy for OCR
            page_np = np.asarray(page_img)
            
            # 1. Extract Text
            extracted_text = process_page_ocr(page_np)
            
            # 2. Extract Images
            extracted_images = extract_visual_elements(page_img, images_out_dir, real_page_idx)
            
            # 3. Merge and Sort
            combined_elements = extracted_text + extracted_images
            combined_elements.sort(key=lambda k: k["y_center"])
            aggregated_data.extend(combined_elements)

        # Memory Cleanup
        del pdf_pages
        del page_np
        gc.collect()

    # Determine Output Filename
    if start_page == 1 and end_page is None:
        doc_name = f"{f_base}.docx"
    else:
        doc_name = f"{f_base}_p{start_page}-{end_page}.docx"
        
    full_doc_path = os.path.join(specific_out_dir, doc_name)
    generate_word_document(aggregated_data, full_doc_path)
    
    print("-" * 50)
    print("WORKFLOW COMPLETED")
    print(f"Document saved to: {full_doc_path}")
    print("-" * 50)


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Medical OCR Extraction Tool")
    
    cli.add_argument("--input", type=str, default=None, help="Path to specific PDF file")
    cli.add_argument("--start", type=int, default=1, help="Start page")
    cli.add_argument("--end", type=int, default=None, help="End page")

    arguments = cli.parse_args()
    
    if arguments.input is None:
        # Batch mode: Scan directory
        target_dir = "input_pdf"
        if os.path.exists(target_dir):
            file_list = [f for f in os.listdir(target_dir) if f.lower().endswith(".pdf")]
            
            if not file_list:
                print(f"No PDF files found in directory '{target_dir}'")
            else:
                print(f"Found {len(file_list)} files. Starting batch processing...")
                for pdf_file in file_list:
                    abs_path = os.path.join(target_dir, pdf_file)
                    orchestrate_pipeline(abs_path, arguments.start, arguments.end)
        else:
            print(f"Directory '{target_dir}' not found. Create it or use --input.")
    else:
        # Single file mode
        orchestrate_pipeline(arguments.input, arguments.start, arguments.end)