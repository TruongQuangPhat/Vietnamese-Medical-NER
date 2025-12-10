import os
import re
import torch
import numpy as np
import gc
import argparse
from typing import List, Dict
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from tqdm import tqdm

# --- IMPORT MODELS ---
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from easyocr import Reader

# --- CONFIGURATION ---
Image.MAX_IMAGE_PIXELS = None

class Config:
    DPI_RESOLUTION = 300 # Giữ 300 để OCR nét nhất
    MARGIN_HEADER = 0.05 # Bỏ 5% trên cùng (Header)
    MARGIN_FOOTER = 0.1  # Bỏ 10% dưới cùng (Footer)
    BATCH_SIZE = 10
    
    # Regex bắt mục lục cấp 2 (VD: 2.1, 6.2...)
    REGEX_HEADER_L2 = re.compile(r"^\s*(\d+(?:\.\d+)+)\.?\s+(?P<content>\S.*)$")

# --- INIT DEVICE ---
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
# 1. UTILS
# ==============================================================================
def add_padding(img_np, pad=10):
    """Thêm viền trắng giúp VietOCR đọc chuẩn hơn"""
    h, w = img_np.shape[:2]
    padded = np.ones((h + 2*pad, w + 2*pad, 3), dtype=np.uint8) * 255
    padded[pad:h+pad, pad:w+pad] = img_np
    return padded

def validate_header(text):
    """Kiểm tra xem dòng text có phải là Header 2.1, 2.2... không"""
    if len(text) < 4: return False
    clean = text.strip()
    
    # Blacklist (tránh nhầm caption ảnh)
    if any(clean.lower().startswith(x) for x in ["Hình", "Hinh", "Bảng", "Bang", "Sơ đồ"]):
        return False

    match = Config.REGEX_HEADER_L2.match(clean)
    if not match: return False
    
    # Chỉ lấy đúng 2 cấp số (VD: 2.1), bỏ 2.1.1
    nums = [n for n in match.group(1).split('.') if n.isdigit()]
    if len(nums) != 2: return False
    
    # Lọc đơn vị đo (1.5 mg)
    content = match.group("content")
    if re.search(r"^(mg|g|ml|lít|cm|mm|%)", content, re.IGNORECASE): return False
    if not re.search(r"[a-zA-Z]", content): return False
    
    return True

# ==============================================================================
# 2. OCR ENGINE (TEXT ONLY)
# ==============================================================================
def ocr_process_page(img_array: np.ndarray) -> List[Dict]:
    img_h, img_w = img_array.shape[:2]
    
    try:
        results = text_detector.readtext(img_array, detail=1, width_ths=0.7)
    except: return []

    raw_blocks = []
    y_min_limit = img_h * Config.MARGIN_HEADER
    y_max_limit = img_h * (1 - Config.MARGIN_FOOTER)

    for box, _, conf in results:
        if conf < 0.25: continue
        
        y_coords = [p[1] for p in box]
        y_center = sum(y_coords) / 4
        
        # Calculate Height of the text box (Quan trọng cho Dynamic Tolerance)
        box_height = max(y_coords) - min(y_coords)

        if y_center < y_min_limit or y_center > y_max_limit: continue

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))

        crop = img_array[max(0, y1-5):min(img_h, y2+5), max(0, x1-5):min(img_w, x2+5)]
        if crop.size == 0: continue

        try:
            padded_crop = add_padding(crop)
            text = text_recognizer.predict(Image.fromarray(padded_crop))
            
            if len(text.strip()) > 1:
                raw_blocks.append({
                    'text': text.strip(),
                    'y_center': y_center / img_h,
                    'x_min': x1 / img_w,
                    'x_max': x2 / img_w,
                    'height': box_height / img_h # Lưu chiều cao chuẩn hóa
                })
        except: continue

    if not raw_blocks: return []

    # --- [FIX LOGIC] DYNAMIC LINE MERGING ---
    raw_blocks.sort(key=lambda item: item['y_center'])
    
    lines = []
    current_line = [raw_blocks[0]]
    
    for i in range(1, len(raw_blocks)):
        current_block = raw_blocks[i]
        prev_block = current_line[-1]
        
        # Tính chiều cao trung bình của dòng đang xét
        avg_height = sum(b['height'] for b in current_line) / len(current_line)
        
        # Dynamic Threshold: 60% chiều cao dòng chữ
        # Nếu dòng cao (tiêu đề), threshold lớn -> Chấp nhận nghiêng
        # Nếu dòng thấp (footnote), threshold nhỏ -> Tách dòng chặt chẽ
        dynamic_threshold = avg_height * 0.6 
        
        diff = abs(current_block['y_center'] - prev_block['y_center'])
        
        if diff <= dynamic_threshold:
            current_line.append(current_block)
        else:
            lines.append(current_line)
            current_line = [current_block]
    lines.append(current_line)

    final_output = []
    for line_group in lines:
        line_group.sort(key=lambda item: item['x_min'])
        joined_text = " ".join([b['text'] for b in line_group])
        
        min_x = line_group[0]['x_min']
        max_x = line_group[-1]['x_max']
        avg_y = sum(b['y_center'] for b in line_group) / len(line_group) # Re-calc avg Y
        
        final_output.append({
            'type': 'text',
            'content': joined_text,
            'y_center': avg_y,
            'x_min': min_x,
            'x_max': max_x
        })
        
    return final_output

# ==============================================================================
# 3. EXPORT TO WORD
# ==============================================================================
def save_to_word(data_list: List[Dict], output_path: str):
    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(13)
    
    buffer = []
    
    def flush(buf):
        for item in buf:
            doc.add_paragraph(item['content'])
            
    for item in data_list:
        text = item['content']
        
        # Kiểm tra Header cấp 2 để ngắt đoạn
        if validate_header(text):
            print(f"[SECTION] {text[:40]}...")
            if buffer:
                flush(buffer)
                # Chèn thẻ Break đỏ
                p = doc.add_paragraph()
                run = p.add_run('</break>')
                run.font.color.rgb = RGBColor(255, 0, 0)
                run.font.bold = True
                p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                
                buffer = []
            buffer.append(item)
        else:
            buffer.append(item)
            
    if buffer: flush(buffer)
    doc.save(output_path)

# ==============================================================================
# 4. MAIN PIPELINE
# ==============================================================================
def run_pipeline(pdf_path, start=1, end=None):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join("output", f"{base_name}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Processing: {base_name}")
    
    try:
        info = pdfinfo_from_path(pdf_path)
        total = info["Pages"]
    except: total = 1000
    
    target_end = min(end, total) if end else total
    all_data = []
    
    # Chạy Batch
    for b_start in range(start, target_end + 1, Config.BATCH_SIZE):
        b_end = min(b_start + Config.BATCH_SIZE - 1, target_end)
        print(f"Batch: {b_start} -> {b_end}")
        
        try:
            images = convert_from_path(pdf_path, dpi=Config.DPI_RESOLUTION, first_page=b_start, last_page=b_end)
        except Exception as e:
            print(f"PDF Error: {e}")
            continue
            
        for img in images:
            # Convert PIL -> Numpy
            img_np = np.asarray(img)
            # OCR Text Only
            page_text = ocr_process_page(img_np)
            all_data.extend(page_text)
            
        del images
        gc.collect()
        
    out_name = f"{base_name}.docx"
    save_path = os.path.join(out_dir, out_name)
    save_to_word(all_data, save_path)
    
    print("-" * 30)
    print(f"DONE: {save_path}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()
    
    if args.input:
        run_pipeline(args.input, args.start, args.end)
    else:
        target = "input"
        if os.path.exists(target):
            for f in os.listdir(target):
                if f.endswith(".pdf"):
                    run_pipeline(os.path.join(target, f), args.start, args.end)