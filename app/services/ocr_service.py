"""
Agri-Mitra — OCR Service
Extracts survey number, owner name, area, and soil type from Pahani/RTC PDFs.
Converts regional land units (Guntha, Bigha, Cent, Acre) to Hectares.
"""
import re
import io
import pymupdf
from PIL import Image
from app.core.config import get_settings

try:
    import pytesseract
    settings = get_settings()
    if settings.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
except Exception:
    pytesseract = None

# Regional unit conversion table (to Hectares)
UNIT_CONVERSIONS: dict[str, float] = {
    "guntha": 0.010117,
    "gunta": 0.010117,
    "bigha": 0.2529,
    "acre": 0.4047,
    "cent": 0.004047,
    "dismil": 0.004047,
    "decimal": 0.004047,
    "kanal": 0.0809,
    "marla": 0.00505,
    "hectare": 1.0,
    "ha": 1.0,
}

UNIT_ALIAS_MAP: dict[str, str] = {
    "acre": "acre",
    "acres": "acre",
    "guntha": "guntha",
    "gunthas": "guntha",
    "gunta": "guntha",
    "guntas": "guntha",
    "bigha": "bigha",
    "bighas": "bigha",
    "cent": "cent",
    "cents": "cent",
    "dismil": "dismil",
    "decimal": "decimal",
    "kanal": "kanal",
    "marla": "marla",
    "hectare": "hectare",
    "hectares": "hectare",
    "ha": "ha",
}

AREA_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(acres?|gunt[ah]a?s?|bighas?|cents?|dismil|decimals?|kanal|marla|hectares?|ha)\b",
    re.IGNORECASE
)

SOIL_TAXONOMY_MAP = {
    "black cotton": "Black Cotton Soil (Vertisol)",
    "regur": "Black Cotton Soil (Vertisol)",
    "kari mannu": "Black Cotton Soil (Vertisol)",
    "red loamy": "Red Loamy (Alfisol)",
    "murrum": "Red Loamy (Alfisol)",
    "kalu murum": "Red Loamy (Alfisol)",
    "alluvial": "Alluvial Soil",
    "delta": "Alluvial Soil",
}


def _pdf_to_images(pdf_bytes: bytes) -> list[Image.Image]:
    """Convert PDF pages to PIL Images for Tesseract OCR."""
    images = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
    return images


def _extract_text(pdf_bytes: bytes) -> str:
    """Run Tesseract or native PyMuPDF text extraction on PDF."""
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            text_pages = [page.get_text() for page in doc]
            combined = "\n".join(text_pages).strip()
            if len(combined) > 50:
                return combined
    except Exception:
        pass

    if pytesseract is not None:
        images = _pdf_to_images(pdf_bytes)
        texts = [pytesseract.image_to_string(img, lang="eng") for img in images]
        return "\n".join(texts)

    return ""


def _parse_area(text: str) -> float | None:
    """
    Parse area in any regional unit from OCR text and convert to Hectares.
    Handles: '4 Acres 12 Guntha', '3 Bigha 10 Guntha', '5 Cents', '1.88 Hectares'
    """
    text_lower = text.lower()

    # Look specifically for Total Area or Extent line first
    target_text = text_lower
    for line in text_lower.splitlines():
        if "total area" in line or "total extent" in line or "gat area" in line:
            if "(" in line:
                line = line.split("(")[0]
            target_text = line
            break

    total_ha = 0.0
    found = False

    for match in AREA_PATTERN.finditer(target_text):
        try:
            value = float(match.group(1))
            unit_raw = match.group(2).lower()
            canonical_unit = UNIT_ALIAS_MAP.get(unit_raw)
            if canonical_unit and canonical_unit in UNIT_CONVERSIONS:
                factor = UNIT_CONVERSIONS[canonical_unit]
                total_ha += value * factor
                found = True
        except (ValueError, KeyError):
            continue

    return round(total_ha, 4) if found else None


def _extract_field(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def parse_pahani(pdf_bytes: bytes) -> dict:
    """
    Main entry point: extracts structured data from Pahani/RTC PDF.
    Returns raw OCR text + parsed fields for human review before DB commit.
    """
    raw_text = _extract_text(pdf_bytes)

    survey_number = _extract_field(raw_text, [
        r"survey\s+(?:number|no\.?)[:\s]+([A-Z0-9/\-]+)",
        r"gat\s+(?:number|no\.?)[:\s]+([A-Z0-9/\-]+)",
        r"khata\s+(?:number|no\.?)[:\s]+([A-Z0-9/\-]+)",
    ])

    owner_name = _extract_field(raw_text, [
        r"owner\s+name[:\s]+([A-Za-z\s\.]+?)(?:\n|phone|father)",
        r"pattadar\s+name[:\s]+([A-Za-z\s\.]+?)(?:\n|phone|mobile)",
    ])

    village = _extract_field(raw_text, [
        r"village[:\s]+([A-Za-z\s]+?)(?:\n|taluk|district|mandal)",
    ])

    district = _extract_field(raw_text, [
        r"district[:\s]+([A-Za-z\s]+?)(?:\n|taluka|mandal|hobli)",
    ])

    total_area_ha = _parse_area(raw_text)

    soil_type = None
    text_lower = raw_text.lower()
    for term, canonical in SOIL_TAXONOMY_MAP.items():
        if term in text_lower:
            soil_type = canonical
            break

    return {
        "raw_ocr_text": raw_text[:3000],
        "parsed": {
            "survey_number": survey_number,
            "owner_name": owner_name,
            "village": village,
            "district": district,
            "total_area_hectares": total_area_ha,
            "detected_soil_type": soil_type,
        },
        "requires_review": True,
        "confidence": "medium" if all([survey_number, owner_name, total_area_ha]) else "low",
    }
