import os
import time

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

PDF = "documents/havacilik.pdf"

# ---- PDF hakkında bilgi ----
from pypdf import PdfReader
reader = PdfReader(PDF)
first_page_text = (reader.pages[0].extract_text() or "").strip()
print(f"Sayfa sayısı: {len(reader.pages)}")
print(f"Metin katmanı var mı: {'EVET' if first_page_text else 'HAYIR (taranmış PDF, OCR gerekir)'}")

try:
    import torch
    print(f"GPU (CUDA) var mı: {torch.cuda.is_available()}")
except ImportError:
    pass
print(f"CPU çekirdek sayısı: {os.cpu_count()}")
print("=" * 60)


def make_converter(do_ocr, do_tables, table_mode=TableFormerMode.ACCURATE):
    options = PdfPipelineOptions(do_ocr=do_ocr, do_table_structure=do_tables)
    options.table_structure_options.mode = table_mode
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


configs = {
    "1) Varsayılan (OCR açık, tablo ACCURATE)": make_converter(True, True),
    "2) OCR kapalı, tablo ACCURATE":            make_converter(False, True),
    "3) OCR kapalı, tablo FAST":                make_converter(False, True, TableFormerMode.FAST),
    "4) OCR kapalı, tablo modeli kapalı":       make_converter(False, False),
}

for name, converter in configs.items():
    converter.convert(PDF)                     # ısınma: modeller yüklensin, ölçüme katılmasın
    start = time.time()
    result = converter.convert(PDF)
    elapsed = time.time() - start
    text_length = len(result.document.export_to_markdown())
    print(f"{name:45} {elapsed:6.1f} sn   ({text_length} karakter metin)")