# Nested NER (Vietnamese COVID-19) — PhoBERT + CRF, Layered BIO

Pipeline này thực hiện Nested NER theo tinh thần paper “Nested Named-Entity Recognition on Vietnamese COVID-19: Dataset and Experiments”, dùng PhoBERT encoder và CRF decoding. Do PhoBERT không có Fast tokenizer (không hỗ trợ `offset_mapping`), ta tự xử lý ở mức từ (word-level) và chỉ tính loss/decoding trên subword-đầu-từ bằng mặt nạ (word_mask).

## 1) Thiết lập môi trường (Windows PowerShell)

```powershell
cd D:\projects

python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell cảnh báo symlink của `huggingface_hub`, có thể bỏ qua. Muốn hết cảnh báo: bật Developer Mode hoặc chạy PowerShell dưới quyền Administrator.

## 2) Tải dữ liệu và chuyển đổi

```powershell
mkdir -p data\raw
Invoke-WebRequest https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word/train_word.conll -OutFile data/raw/train.conll
Invoke-WebRequest https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word/dev_word.conll   -OutFile data/raw/dev.conll
Invoke-WebRequest https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word/test_word.conll  -OutFile data/raw/test.conll

python scripts\convert_conll_to_jsonl.py --input data\raw\train.conll --output data\train.jsonl
python scripts\convert_conll_to_jsonl.py --input data\raw\dev.conll   --output data\dev.jsonl
python scripts\convert_conll_to_jsonl.py --input data\raw\test.conll  --output data\test.jsonl
```

Kỳ vọng log:
- Wrote data/train.jsonl (≈5027 examples)
- Wrote data/dev.jsonl (≈2000 examples)
- Wrote data/test.jsonl (≈3000 examples)

## 3) Train (PhoBERT + CRF)

```powershell
python -m src.train_layered `
  --train data\train.jsonl `
  --dev data\dev.jsonl `
  --outdir runs\phoner_phobert_crf `
  --pretrained vinai/phobert-base `
  --max_layers 2 `
  --epochs 10 `
  --batch_size 16 `
  --lr 5e-5 `
  --warmup_ratio 0.1 `
  --max_length 256
```

Gợi ý:
- Thiếu VRAM: giảm `--batch_size 8` hoặc `4`.
- Checkpoint: `runs\phoner_phobert_crf\best.pt`.

## 4) Evaluate (đang tối giản: tính loss dev trong train). Nếu cần F1 strict span-level, báo lại để bổ sung script chi tiết.

python -m src.evaluate_nested --model runs\phoner_phobert_crf\best.pt --data data\test.jsonl --max_length 256 --out results\eval_test_best.json --pred_out results\pred_test_best.jsonl

python -m src.evaluate_nested --model runs\phoner_phobert_crf\final.pt --data data\test.jsonl --max_length 256 --out results\eval_test_final.json

## 5) Predict nhanh

```powershell
python -m src.predict_nested --model runs\phoner_phobert_crf\best.pt --text "Bệnh nhân COVID-19 điều trị bằng remdesivir tại Hà Nội." --max_length 256
```

## Ghi chú kỹ thuật

- Gán nhãn BIO ở cấp từ (word-level) theo các span ký tự vàng; khi token hóa bằng PhoBERT "slow", ta dàn sang subword và dùng `word_mask` để chỉ tính tại subword đầu từ trong CRF.
- Nested NER xử lý theo “layered BIO”: xếp thực thể vào tối đa `--max_layers` lớp không chồng lấn, mỗi lớp một CRF head.

Thư mục khuyến nghị:
```
data/
runs/
scripts/
  convert_conll_to_jsonl.py
src/
  __init__.py
  data_nested.py
  predict_nested.py
  train_layered.py
  models/
    __init__.py
    layered_ner_crf.py
requirements.txt
README.md
```