import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import io

class OCREngine:
    @staticmethod
    def extract_text(file_bytes: bytes, file_name: str) -> str:
        """Extracts text from scanned image or PDF file bytes without caching state."""
        ext = file_name.split('.')[-1].lower()
        extracted_text = ""

        try:
            if ext in ['png', 'jpg', 'jpeg']:
                image = Image.open(io.BytesIO(file_bytes))
                extracted_text = pytesseract.image_to_string(image)
                
            elif ext == 'pdf':
                images = convert_from_bytes(file_bytes)
                pages_text = [pytesseract.image_to_string(img) for img in images]
                extracted_text = "\n".join(pages_text)
                
            elif ext == 'txt':
                extracted_text = file_bytes.decode('utf-8')
            else:
                raise ValueError(f"Unsupported file format: .{ext}")

        except Exception as e:
            raise RuntimeError(f"OCR Extraction failed for {file_name}: {str(e)}")

        cleaned = "\n".join([line.strip() for line in extracted_text.splitlines() if line.strip()])
        return cleaned