Set-Location 'D:\projects'
if (Test-Path '.\.venv\Scripts\Activate.ps1') { . '.\.venv\Scripts\Activate.ps1' } else { Write-Warning 'KhÃ´ng tÃ¬m tháº¥y venv: .\.venv\Scripts\Activate.ps1' }
cd D:\projects

python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
New-Item -ItemType Directory -Force -Path data\raw | Out-Null

Invoke-WebRequest https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word/train_word.conll -OutFile data/raw/train.conll
Invoke-WebRequest https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word/dev_word.conll   -OutFile data/raw/dev.conll
Invoke-WebRequest https://raw.githubusercontent.com/VinAIResearch/PhoNER_COVID19/main/data/word/test_word.conll  -OutFile data/raw/test.conll

python scripts\convert_conll_to_jsonl.py --input data\raw\train.conll --output data\train.jsonl
python scripts\convert_conll_to_jsonl.py --input data\raw\dev.conll   --output data\dev.jsonl
python scripts\convert_conll_to_jsonl.py --input data\raw\test.conll  --output data\test.jsonl
python -m src.train_layered --train data\train.jsonl --dev data\dev.jsonl --outdir runs\phoner_phobert_crf --pretrained vinai/phobert-base --max_layers 2 --epochs 10 --batch_size 16 --lr 5e-5 --warmup_ratio 0.1 --max_length 256
python -m pip install --force-reinstall --no-cache-dir torchcrf
python -c "import sys; print(sys.executable); import torchcrf; print('torchcrf OK at:', torchcrf.__file__)"
python -m pip uninstall -y torchcrf pytorch-crf
python -m pip install -U pytorch-crf
python -c "import sys; print(sys.executable); import torchcrf; print('torchcrf OK at:', torchcrf.__file__)"
python -m src.train_layered --train data\train.jsonl --dev data\dev.jsonl --outdir runs\phoner_phobert_crf --pretrained vinai/phobert-base --max_layers 2 --epochs 10 --batch_size 16 --lr 5e-5 --warmup_ratio 0.1 --max_length 256
python -m src.train_layered --train data\train.jsonl --dev data\dev.jsonl --outdir runs\phoner_phobert_crf --pretrained vinai/phobert-base --max_layers 2 --epochs 10 --batch_size 16 --lr 5e-5 --warmup_ratio 0.1 --max_length 256
python -c "from src.data_nested import load_jsonl, build_label_list_from_data, assign_layers, encode_bio_layers_phobert; print('OK')"
python -m src.train_layered --train data\train.jsonl --dev data\dev.jsonl --outdir runs\phoner_phobert_crf --pretrained vinai/phobert-base --max_layers 2 --epochs 10 --batch_size 16 --lr 5e-5 --warmup_ratio 0.1 --max_length 256
python -m src.evaluate_nested --model runs\phoner_phobert_crf\best.pt --data data\test.jsonl --max_length 256 --out results\eval_test_best.json --pred_out results\pred_test_best.jsonl

python -m src.evaluate_nested --model runs\phoner_phobert_crf\final.pt --data data\test.jsonl --max_length 256 --out results\eval_test_final.json
python -m src.predict_nested --model runs\phoner_phobert_crf\best.pt --text "Bệnh nhân COVID-19 điều trị bằng remdesivir tại Hà Nội." --max_length 256
python -m src.predict_nested --model runs\phoner_phobert_crf\best.pt --text "Bệnh nhân COVID-19 điều trị bằng remdesivir tại Hà Nội." --max_length 256
python - <<'PY'\nimport json; print(json.loads(open('data/dev.jsonl', encoding='utf-8').readline())['text'])\nPY
python -c "import json; print(json.loads(open(r'data/dev.jsonl', encoding='utf-8').readline())['text'])"
$txt = 'Bác_sĩ Nguyễn_Trung_Nguyên , Giám_đốc Trung_tâm Chống độc , Bệnh_viện Bạch_Mai , cho biết bệnh_nhân được chuyển đến bệnh_viện ngày 7/3 , chẩn_đoán ngộ_độc thuốc điều_trị sốt_rét chloroquine .'
python -m src.predict_nested --model runs\phoner_phobert_crf\best.pt --text "$txt" --max_length 256
python -m src.predict_nested --model runs\phoner_phobert_crf\best.pt --text "$txt" --max_length 256 --out results\pred_example.json
New-Item -ItemType Directory -Force -Path results, backup | Out-Null
Copy-Item runs\phoner_phobert_crf\best.pt backup\best.pt
Copy-Item runs\phoner_phobert_crf\final.pt backup\final.pt
Copy-Item runs\phoner_phobert_crf\labels.json backup\labels.json
pip freeze | Set-Content -Encoding UTF8 backup\requirements.lock.txt
.\scripts\predict_5_examples.ps1
python scripts\predict_5_examples.py
python scripts\eval_to_md.py
python scripts\pred_to_md.py results\pred_examples_YYYYMMDD_HHmm.jsonl results\pred_examples.md
python scripts\pred_to_md.py "results\pred_examples_*.jsonl" results\pred_examples.md
New-Item -ItemType Directory -Force -Path results | Out-Null
$content = Get-Clipboard
$matches = [regex]::Matches($content, '\[ep\s*(\d+)\]\s*dev_loss=([0-9.]+)')
$rows = foreach ($m in $matches) {
  [pscustomobject]@{
    epoch    = [int]$m.Groups[1].Value
    dev_loss = [double]$m.Groups[2].Value
  }
}
$rows | Sort-Object epoch | Export-Csv -NoTypeInformation results\metrics.csv
Get-Content results\metrics.csv
pip install matplotlib
python scripts\plot_training_curve.py results\metrics.csv
python scripts\plot_training_curve.py --csv results\metrics.csv --out results\loss.png --svg results\loss.svg --smooth 1
python scripts\plot_training_curve.py --csv results\metrics.csv --out results\loss.png --svg results\loss.svg --smooth 1
python scripts\plot_training_curve.py --csv results\metrics.csv --out results\loss.png --svg results\loss.svg --smooth 1
python scripts\plot_training_curve.py --csv results\metrics.csv --out results\loss.png --svg results\loss.svg --smooth 1
python scripts\plot_training_curve.py --csv results\metrics.csv --out results\loss.png --svg results\loss.svg --smooth 1
python scripts\plot_training_curve.py --csv results\metrics.csv --out results\loss.png --svg results\loss.svg --smooth 1
# Đổi đường dẫn cho phù hợp
Set-Location 'D:\projects'
# Kích hoạt venv (đường dẫn tuyệt đối)
. 'D:\projects\.venv\Scripts\Activate.ps1'
notepad $PROFILE
Save-ProjectSession -ProjectRoot "D:\projects" -Path "D:\projects\replay.ps1" -Last ((Get-History).Count)
notepad $PROFILE
Save-ProjectSession -ProjectRoot "D:\projects" -Path "D:\projects\replay.ps1" -All
Save-ProjectSession -ProjectRoot "D:\projects" -Path "D:\projects\replay.ps1" -Last ((Get-History).Count)
if (!(Test-Path $PROFILE)) { $dir = Split-Path $PROFILE; if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force }; New-Item -ItemType File -Path $PROFILE -Force }
notepad $PROFILE
. $PROFILE
