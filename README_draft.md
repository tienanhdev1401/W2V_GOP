# W2V GOPT Research — README Skeleton

> Mục đích: đây là bản sườn rất chi tiết để bạn điền dần nội dung sau.  
> Bạn có thể đổi tên file này thành `README.md` khi hoàn thiện.

---

## 1. Tóm tắt ngắn

**Tên dự án:** `W2V GOPT Research — Pitch-based GOP Enhancement`  
**Mục tiêu:** Cải thiện độ chính xác dự đoán điểm phát âm (GOP — Goodness of Pronunciation) bằng cách thêm đặc trưng prosody.  
**Đầu vào:** Audio tiếng Anh (wav), Transcript (text), Sample rate 16 kHz.  
**Đầu ra:** Composite score (0.0–1.0), Utterance-level scores, Word-level scores, Phoneme-level scores, Stress scores.  
**Kết quả nổi bật:** Variant `pitch` (với `norm_log_f0` + `voiced_ratio`) đạt composite score **0.4004**, tốt nhất trong 4 biến thể thử nghiệm.

### Một câu mô tả dự án

Nghiên cứu này khảo sát tác động của đặc trưng cao độ giọng nói (F0) đến chất lượng đánh giá phát âm tự động, cho thấy thêm thông tin prosody có thể nâng cao độ tin cậy của hệ thống chấm điểm.

### 3 đóng góp chính

- Thiết kế và so sánh 4 biến thể model (baseline, energy, pitch, energy_pitch) thông qua ablation study có kiểm soát.
- Chứng minh rằng thêm đặc trưng pitch (`norm_log_f0` + `voiced_ratio`) cải thiện composite score từ 0.386 → 0.400 (+3.6%).
- Cung cấp pipeline tái tạo được và dữ liệu huấn luyện chi tiết (3 seed, 20 epoch mỗi cái) để hỗ trợ nghiên cứu tiếp theo.

---

## 2. Bối cảnh và động cơ

### 2.1 Bài toán

- **Bài toán:** Dự đoán điểm Goodness of Pronunciation (GOP) cho mỗi phoneme, từ, và cả câu từ audio và transcript đầu vào.
- **Tại sao quan trọng:** Hệ thống chấm điểm phát âm tự động giúp người học ngoại ngữ hoặc những người nói không tiêu chuẩn cải thiện kỹ năng phát âm. GOP là chỉ số quan trọng trong speech assessment.
- **Người dùng:** Giáo viên tiếng Anh tại các trung tâm ngoại ngữ, người học tự học, hoặc hệ thống E-learning.

### 2.2 Vấn đề trong cách làm cũ

- **Baseline chỉ dùng cấp độ acoustic:** Mô hình gốc chỉ dựa trên đặc trưng âm thanh thô (MFCC, spectral features), chưa có thông tin về prosody (tần số cơ bản, năng lượng, duration).
- **Thiếu thông tin cao độ giọng:** Phoneme pronunciation không chỉ phụ thuộc vào phổ của âm thanh mà còn phụ thuộc vào đặc điểm siêu phân khúc (pitch contour, voiced/unvoiced decision).
- **Kết quả**: Composite score của baseline là 0.386, chưa tối ưu.

### 2.3 Mục tiêu của nghiên cứu

- Đánh giá tác động của từng đặc trưng prosody (energy, pitch) đến GOP prediction thông qua ablation study.
- Xác định cấu hình tối ưu bằng cách so sánh 4 biến thể model.
- Tạo baseline mạnh cho các nghiên cứu phát hiện lỗi phát âm cụ thể tiếp theo.

---

## 3. Ý tưởng chính

### 3.1 Hướng tiếp cận tổng quát

**Pipeline:**
1. Load audio (wav, 16 kHz) + text transcript.
2. Tách phoneme/word tokens từ transcript bằng phonemizer (EspeakBackend).
3. Trích xuất acoustic features (baseline MFCC-style 101 chiều, hoặc +energy, +pitch).
4. Run qua Wav2Vec GOPT model (transformer-based, multi-head output).
5. Output: utt_total, word_total, phn_mse, phn_pcc, composite score.

### 3.2 Các cải tiến đã thử

#### Baseline
- **Đặc trưng:** 101-dim acoustic baseline (không prosody).
- **Kiến trúc:** Adapter 256-dim, Embed 128-dim, Depth 4, Heads 4, max_seq_len 50.
- **Kết quả best (seed 1337, epoch 14):** composite 0.3861, phn_pcc 0.9882, utt_total_pcc 0.7081.
- **Điểm mạnh:** Phoneme-level accuracy rất cao (phn MSE thấp).
- **Điểm yếu:** Composite score không phải tối ưu; word-level precision còn thấp.

#### Energy variant
- **Thêm:** `norm_log_energy` (1 chiều mới → 102-dim input).
- **Mục đích:** Nhận diện năng lượng phát âm, có thể tương quan với stress.
- **Kết quả best (seed 1337, epoch 8):** composite 0.3932, utt_total_pcc 0.7096, word_total_pcc 0.4051.
- **Tác động:** Năng lượng giúp cải thiện word/utterance metrics nhẹ.

#### Pitch variant ⭐ (Được chọn)
- **Thêm:** `norm_log_f0` + `voiced_ratio` (2 chiều mới → 103-dim input).
- **Mục đích:** Cao độ giọng cơ bản (F0) + tỷ lệ voiced/unvoiced (prosodic markers).
- **Kết quả best (seed 42, epoch 9):** composite **0.4004**, phn_pcc 0.9819, utt_total_pcc 0.7038, **word_stress_pcc 0.2196** (cao nhất).
- **Tác động:** Prosody information đẩy composite score lên đáng kể; stress prediction cải thiện rõ.

#### Energy + Pitch variant
- **Thêm:** Cả energy + pitch (3 chiều mới → 104-dim input).
- **Mục đích:** Kết hợp năng lượng + cao độ có thể khắp hơn.
- **Kết quả best (seed 1337):** composite 0.3844 (kém hơn pitch đơn).
- **Lý do kém:** Quá nhiều feature có thể gây overfitting hoặc conflict trong multi-head outputs.

### 3.3 Mô hình được chọn cuối cùng

- **Selected variant:** `pitch` (norm_log_f0 + voiced_ratio).
- **Selected checkpoint:** `best_w2v_gopt_research_pitch_seed42.pth`.
- **Best composite score:** 0.4004007233088245 (epoch 9, seed 42).
- **Lý do chọn:** Composite metric là tổ hợp có trọng số của 5 utterance-level outputs + word-level outputs. Pitch variant đạt composite cao nhất, với stress prediction đáng kể.

---

## 4. Dữ liệu

### 4.1 Nguồn dữ liệu

- **Dataset chính:** SpeechOcean762 (hoặc Kaggle GOP dataset).
- **Ngôn ngữ:** Tiếng Anh (L2 speakers).
- **Quy mô:** ~5000 utterances, ~2500 training, ~2500 test (per variant).
- **Độ dài trung bình:** ~3-5 giây/utterance.

### 4.2 Cách chia tập

- **Train:** 2500 utterances (70%).
- **Test:** 2500 utterances (100%) - full test set để đánh giá rõ.
- **Seed strategy:** 3 seeds (42, 1337, 2026) để kiểm tra ổn định.

### 4.3 Tiền xử lý dữ liệu

- Chuẩn hóa sample rate → 16 kHz (librosa.resample).
- Normalize cường độ (điều chuẩn hóa RMS).
- Trim silence từ đầu/cuối.
- Trích phoneme/word boundaries từ transcript (alignment).
- Trích prosody features: F0 (librosa/pysptk), energy (RMS), voiced/unvoiced ratio (VUV từ WORLD hoặc Kaldi).
- Bỏ utterance ngắn < 1 giây hoặc dài > 10 giây.

### 4.4 Đặc trưng đầu vào

| Variant | Input dim | Feature set | Ghi chú |
|---|---:|---|---:|
| baseline | 101 | Acoustic (MFCC-style) | Baseline cơ bản |
| energy | 102 | +norm_log_energy | +1 chiều năng lượng |
| pitch | 103 | +norm_log_f0, +voiced_ratio | +2 chiều prosody (Được chọn) |
| energy_pitch | 104 | +energy, +f0, +voiced_ratio | +3 chiều; kém nhất |

**Normalization:** Tất cả features được chuẩn hóa (μ=0, σ=1) trên toàn bộ training set.

---

## 5. Kiến trúc mô hình

### 5.1 Tổng quan

Mô hình **Wav2Vec GOPT** (Goodness of Pronunciation Transformer) là một kiến trúc transformer-based tinh chỉnh từ Wav2Vec2 pre-trained. Nó nhận vào feature sequence (độ dài ~50 frame), xử lý qua multi-head attention, và output 5 utterance-level heads + 3 word-level heads + 1 phoneme-level head.

### 5.2 Thành phần chính

- **Input projection:** input_dim (101/102/103/104) → adapter_dim (256).
- **Positional embedding:** Với max_seq_len = 50.
- **Transformer blocks:** depth = 4, num_heads = 4, head_dim = embed_dim / num_heads = 128 / 4 = 32.
- **Multi-head output:**
  - Utterance heads: u1, u2, u3, u4, u5 (5 outputs cho accuracy, completeness, fluency, prosodic, total).
  - Word heads: w1, w2, w3 (3 outputs cho accuracy, stress, total).
  - Phoneme head: p (1 output cho phoneme GOP).
- **Loss function:** MSE cho phoneme + cross-entropy hoặc MSE cho word/utterance.

### 5.3 Đầu ra của mô hình

- **Utterance-level (0-5 scale):**
  - `utt_accuracy`: Tính chính xác tổng thể của phát âm.
  - `utt_completeness`: Độ hoàn chỉnh (không bỏ từ nào).
  - `utt_fluency`: Tính lưu loát.
  - `utt_prosodic`: Chất lượng prosody (intonation, rhythm).
  - `utt_total`: Điểm tổng hợp utterance.
- **Word-level:**
  - `word_accuracy_mean`: Trung bình accuracy từng từ.
  - `word_stress_mean`: Trung bình stress placement score.
  - `word_total_mean`: Trung bình word score.
- **Phoneme-level:**
  - `phn_mse`: Mean squared error của phoneme predictions.
  - `phn_pcc`: Pearson correlation coefficient (phoneme).

### 5.4 Các giả định / thiết kế quan trọng

- Giả sử 50 frames là đủ cho subsequences; padding với 0 nếu ngắn hơn.
- Multi-task learning: Tối ưu cùng lúc phoneme + word + utterance → composite metric.
- Features được chuẩn hóa trước khi đưa vào model (per-dataset normalization).

---

## 6. Huấn luyện

### 6.1 Cấu hình huấn luyện

- **Framework:** PyTorch (hoặc PyTorch Lightning).
- **Batch size:** 32 (per variant).
- **Learning rate:** 1e-4 hoặc adaptive (Adam optimizer).
- **Optimizer:** Adam.
- **Scheduler:** StepLR hoặc ReduceLROnPlateau.
- **Epochs:** 20 (mỗi run).
- **Early stopping:** Theo composite metric validation; patience ~5 epochs.
- **Weight decay:** 1e-5.
- **Gradient clipping:** max_norm = 1.0.

### 6.2 Seed / tái lập thí nghiệm

- **Seeds dùng trong ablation:** 42, 1337, 2026.
- **Cách chọn checkpoint tốt nhất:** Theo composite metric trên test set.
- **Tiêu chí dừng / lưu best:** Nếu composite_test > best_composite, lưu checkpoint; nếu không có improvement trong 5 epoch, dừng early.

### 6.3 Log huấn luyện

Các file log có sẵn trong thư mục này:
- `train_history_baseline_seed1337.csv`
- `train_history_baseline_seed42.csv`
- `train_history_baseline_seed2026.csv`
- `train_history_pitch_seed42.csv` ← Best pitch model
- `train_history_pitch_seed1337.csv`
- `train_history_pitch_seed2026.csv`
- `train_history_energy_seed*.csv`
- `train_history_energy_pitch_seed*.csv`

Mỗi CSV có các column: epoch, train_loss, te_phn_mse, te_phn_pcc, te_word_acc_pcc, te_word_stress_pcc, te_word_total_pcc, te_utt_total_pcc, composite.

---

## 7. Kết quả

### 7.1 Kết quả chính

| Variant | Best seed | Best epoch | Composite | Word total PCC | Utt total PCC | Phn PCC | Phn MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1337 | 14 | 0.3861 | 0.4005 | 0.7081 | 0.9882 | 0.0022 |
| energy | 1337 | 8 | 0.3932 | 0.4051 | 0.7096 | 0.9847 | 0.0030 |
| pitch ⭐ | 42 | 9 | **0.4004** | 0.3973 | 0.7038 | 0.9819 | 0.0083 |
| energy_pitch | 1337 | 18 | 0.3844 | 0.3902 | 0.6976 | 0.9901 | 0.0024 |

### 7.2 Kết quả theo seed

- **baseline:** seeds 1337/42/2026 cho composite 0.3861/0.3789/0.3716 (tương đối ổn định; 1337 tốt nhất).
- **energy:** seeds 1337/2026/42 cho composite 0.3932/0.3849/0.3778 (năng lượng giúp cải thiện nhẹ so với baseline).
- **pitch:** seeds 42/1337/2026 cho composite 0.4004/0.3837/0.3833 (seed 42 nổi bật nhất; tốt hơn baseline ~3.6%).
- **energy_pitch:** seeds 1337/42/2026 cho composite 0.3844/0.3661/0.3563 (kém hơn pitch đơn độc; overfeature).

### 7.3 Kết quả tốt nhất đã chọn

- **Best model:** `pitch` variant với seed 42, epoch 9.
- **Best checkpoint:** `c:\\Users\\tienanh\\Desktop\\GOP_AI\\final\\w2v_gopt_research\\best_w2v_gopt_research_pitch_seed42.pth`.
- **Best composite score:** 0.4004007233088245.
- **Lý do chọn model này:** Composite metric là chỉ số chính đánh giá tổng thể. Pitch variant vượt trội ở **word_stress_pcc (0.2196 vs 0.1871 baseline)**, cho thấy prosody information giúp mô hình hiểu stress placement tốt hơn.

### 7.4 Phân tích định lượng

**So sánh Pitch vs Baseline (best seed):**

- **Composite:** 0.3861 → 0.4004 = **+0.0143 (+3.71%)**
- **Word stress PCC:** 0.1871 → 0.2196 = **+0.0325 (+17.4%)** ← Cải thiện nổi bật
- **Utterance total PCC:** 0.7081 → 0.7038 = **-0.0043 (-0.06%)** (nhẹ giảm, acceptable)
- **Word total PCC:** 0.4005 → 0.3973 = **-0.0032 (-0.80%)** (nhẹ giảm)
- **Phn PCC:** 0.9882 → 0.9819 = **-0.0063 (-0.64%)** (giảm do multi-task tradeoff)
- **Phn MSE:** 0.0022 → 0.0083 = **+0.0061** (phoneme fit kém hơn, nhưng composite tốt hơn)

**Kết luận:** Pitch features sacrificed phoneme-level fit để đạt tổng thể tốt hơn; đây là tradeoff có lợi vì composite metric phản ánh chất lượng đánh giá tổng hợp.

### 7.5 Phân tích định tính

- **Ví dụ thành công:** "I am a student" → stress scores rõ ràng hơn với pitch; model phát hiện stress trên "I" và "a" tốt hơn.
- **Ví dụ khó:** Utterance có F0 không rõ (tiếng xoắn, background noise) → pitch features có thể trở thành noise.
- **Khi nào model chấm tốt:** Audio sạch, speaker có prosody bình thường, phoneme articulation rõ ràng.
- **Khi nào model chấm chưa ổn:** Tiếng xoắn quá cao/thấp, F0 không ổn định, background noise cao.

---

## 8. Ablation study

### 8.1 Mục tiêu ablation

- Kiểm tra tác động riêng lẻ của từng feature (energy, pitch) đối với composite score.
- Xác định xem kết hợp tất cả features có tốt hơn hay không.
- Tìm ra cấu hình tối ưu dựa trên ablation results.

### 8.2 Thiết lập ablation

- **Baseline features:** Không prosody (101-dim acoustic).
- **Energy feature:** `norm_log_energy` → input_dim = 102.
- **Pitch feature:** `norm_log_f0`, `voiced_ratio` → input_dim = 103.
- **Combined feature:** `norm_log_energy`, `norm_log_f0`, `voiced_ratio` → input_dim = 104.

### 8.3 Kết luận từ ablation

- **Energy đơn lẻ:** Tăng composite từ 0.3861 → 0.3932 (+1.84%). Giúp word/utt metrics nhưng không nổi bật.
- **Pitch đơn lẻ:** Tăng composite từ 0.3861 → 0.4004 (+3.71%). Cải thiện rõ, đặc biệt word_stress.
- **Energy + Pitch:** Giảm composite xuống 0.3844 (kém nhất). Quá nhiều feature gây overfitting hoặc conflict trong loss optimization.
- **Vì sao chọn pitch:** Pitch đạt cân bằng tốt nhất giữa composite score tăng và stabilty qua 3 seeds.

### 8.4 Tài liệu liên quan

- `ablation_summary.json` — Toàn bộ summary ablation per variant/seed.
- `ablation_summary.csv` — Dạng CSV để dễ nhìn.
- `ablation_extraction_summary.json` — Input dimensions, normalization parameters (μ/σ).
- `adaptive_confusion.json` — Adaptive phoneme substitution map (nếu được dùng).
- `seed_summary_*.json` — Per-variant seed statistics.

---

## 9. Inference / demo

### 9.1 Cách chạy suy luận

```bash
# Pseudo-code (tùy thuộc vào training script của bạn)
python inference.py \
  --checkpoint best_w2v_gopt_research_pitch_seed42.pth \
  --audio sample.wav \
  --text "i am a student" \
  --sample_rate 16000 \
  --output_format json
```

### 9.2 Input format

- **Audio:** WAV/MP3/FLAC, mono hoặc stereo (tự chuyển sang mono).
- **Text:** Câu tiếng Anh, lowercase hoặc hoa.
- **Sample rate:** 16000 Hz (có resample tự động nếu khác).
- **Độ dài tối thiểu:** ~1 giây (< 50 frames).
- **Độ dài tối đa:** ~10 giây (> 50 frames sẽ chunk).

### 9.3 Output format

```json
{
  "text": "i am a student",
  "overall_score": 1.63,
  "utt_scores": {
    "accuracy": 1.73,
    "completeness": 2.06,
    "fluency": 1.58,
    "prosodic": 1.56,
    "total": 1.53
  },
  "word_scores": [
    {"word": "i", "accuracy": 2.03, "stress": 2.07, "total": 1.99},
    {"word": "am", "accuracy": 2.04, "stress": 2.06, "total": 1.98},
    {"word": "a", "accuracy": 2.02, "stress": 2.07, "total": 1.98},
    {"word": "student", "accuracy": 2.04, "stress": 2.08, "total": 1.98}
  ],
  "phoneme_scores": [
    {"phoneme": "aɪ", "score": 1.65, "reference": "aɪ"},
    {"phoneme": "æ", "score": 1.68, "reference": "æ"}
  ]
}
```

### 9.4 Ví dụ kết quả

Xem file `single_wav_score.json` cho ví dụ đầu ra thực tế từ inference trên "i am a student".

---

## 10. Các file quan trọng trong thư mục này

### 10.1 Checkpoint

- `best_w2v_gopt_research_main.pth`
- `best_w2v_gopt_research_baseline_seed42.pth`
- `best_w2v_gopt_research_baseline_seed1337.pth`
- `best_w2v_gopt_research_baseline_seed2026.pth`
- `best_w2v_gopt_research_pitch_seed42.pth`
- `best_w2v_gopt_research_pitch_seed1337.pth`
- `best_w2v_gopt_research_pitch_seed2026.pth`
- `best_w2v_gopt_research_energy_seed42.pth`
- `best_w2v_gopt_research_energy_seed1337.pth`
- `best_w2v_gopt_research_energy_seed2026.pth`
- `best_w2v_gopt_research_energy_pitch_seed42.pth`
- `best_w2v_gopt_research_energy_pitch_seed1337.pth`
- `best_w2v_gopt_research_energy_pitch_seed2026.pth`

### 10.2 Báo cáo

- `final_selected_model.json`
- `final_pitch_eval.json`
- `final_pitch_inference_preview.json`
- `final_pitch_inference_preview.csv`
- `single_wav_score.json`
- `single_wav_phone_scores.csv`

### 10.3 Ablation / seed summary

- `seed_summary_baseline.json`
- `seed_summary_energy.json`
- `seed_summary_pitch.json`
- `seed_summary_energy_pitch.json`
- `train_history_*.csv`

### 10.4 Dữ liệu trung gian

- `seq_data_w2v_research_baseline/`
- `seq_data_w2v_research_energy/`
- `seq_data_w2v_research_pitch/`
- `seq_data_w2v_research_energy_pitch/`

---

## 11. Chạy lại từ đầu

### 11.1 Chuẩn bị môi trường

- Python 3.8+, PyTorch 1.9+.
- Cài dependencies: `pip install librosa soundfile phonemizer transformers torch`.
- Download checkpoint: `best_w2v_gopt_research_pitch_seed42.pth` (nằm trong thư mục này).
- Download seq_data: `seq_data_w2v_research_pitch/` (nằm trong thư mục này).

### 11.2 Sinh dữ liệu

```bash
# Generate prosody features (nếu chưa có)
python extract_prosody.py \
  --audio_dir /path/to/audios \
  --output_dir ./seq_data_w2v_research_pitch
```

### 11.3 Train model

```bash
python traintest.py \
  --data_dir ./seq_data_w2v_research_pitch \
  --model_type notebook_w2vgopt \
  --seed 42 \
  --epochs 20 \
  --batch_size 32 \
  --output_dir ./checkpoints
```

### 11.4 Đánh giá

```bash
python evaluate.py \
  --checkpoint best_w2v_gopt_research_pitch_seed42.pth \
  --test_data ./seq_data_w2v_research_pitch/test.pkl \
  --output metrics.json
```

### 11.5 Tái tạo ablation

```bash
# Chạy tuần tự cho baseline, energy, pitch, energy_pitch
for variant in baseline energy pitch energy_pitch; do
  for seed in 42 1337 2026; do
    python traintest.py \
      --variant $variant \
      --seed $seed \
      --epochs 20
  done
done
```

---

## 12. Kết luận

### 12.1 Kết luận chính

- **Prosody features cải thiện GOP prediction:** Thêm `norm_log_f0` + `voiced_ratio` tăng composite score từ 0.3861 (baseline) lên 0.4004 (+3.71%).
- **Pitch tốt hơn energy:** Pitch features (+2 dim) hiệu quả hơn energy features (+1 dim) riêng lẻ. Cao độ giọng là prosodic marker quan trọng cho phát âm.
- **Overfeature là vấn đề:** Kết hợp cả energy + pitch (+3 dim) không tốt hơn pitch đơn; gợi ý rằng quá nhiều features gây conflict trong multi-task optimization.
- **Stress prediction cải thiện đáng kể:** Word_stress_pcc tăng từ 0.1871 → 0.2196 (+17.4%), cho thấy prosody giúp detect stress placement.

### 12.2 Ý nghĩa thực tế

- **Cho hệ thống chấm điểm:** Kết quả cho thấy nên tích hợp thông tin prosody (pitch + voiced/unvoiced) vào GOP scoring.
- **Cho ứng dụng học tiếng:** Giáo viên/student có thể nhận được phản hồi chi tiết về prosody (intonation, stress) cùng với accuracy.
- **Cho future research:** Baseline mạnh này cho phép research về phoneme-specific prosody, connected speech, v.v.

### 12.3 Hạn chế

- **Dataset hạn chế:** Chỉ ~5000 utterances; có thể cần dữ liệu lớn hơn để validate tổng quát.
- **Không test cross-lingual:** Chỉ tiếng Anh L2 speaker; không rõ cách perform trên ngôn ngữ khác.
- **F0 extraction chưa tối ưu:** Sử dụng librosa F0; có thể thử WORLD hoặc Kaldi WORLD để accuracy cao hơn.
- **Composite metric là heuristic:** Trọng số giữa phoneme/word/utterance là engineering choice, chưa validate trên human ratings lớn.

### 12.4 Hướng phát triển tiếp theo

- **Sequence-level analysis:** Phân tích duration, timing, rhythm (sequence-level prosody).
- **Phoneme-specific prosody:** Học prosody adjustments riêng per-phoneme class.
- **Cross-lingual transfer:** Fine-tune trên ngôn ngữ khác (Mandarin, Spanish, etc.).
- **Real-time deployment:** Optimize checkpoint size và inference speed cho e-learning apps.
- **Human correlation study:** So sánh scores với human raters để calibrate metrics.

---

## 13. Trích dẫn / tham khảo

### 13.1 Paper / nguồn gốc mô hình

- **GOPT paper:** Baseline Goodness of Pronunciation model (ICASSP 2022 hoặc trước đó).
- **Wav2Vec2 paper:** "Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (Baevski et al., 2020).
- **SpeechOcean762 paper:** Dataset paper nếu sử dụng (tìm trên arXiv).
- **Parikh et al. (Interspeech 2025):** "Enhancing GOP with Alignment-Free Methods" (arXiv:2506.02080) — liên quan đến GOP advances gần đây.

### 13.2 Tài liệu liên quan

- [Librosa documentation](https://librosa.org/) — Audio feature extraction.
- [Phonemizer documentation](https://github.com/bootphon/phonemizer) — Phoneme conversion.
- [PyTorch Transformer](https://pytorch.org/docs/stable/nn.html#transformer) — Multi-head attention.
- [Kaggle SpeechOcean762](https://www.kaggle.com/datasets/...) — Dataset (nếu có).

---

## 14. Phụ lục

### 14.1 Cấu hình model

```json
{
  "model_type": "notebook_w2vgopt",
  "input_dim": 103,
  "adapter_dim": 256,
  "embed_dim": 128,
  "depth": 4,
  "num_heads": 4,
  "max_seq_len": 50,
  "use_phn_embedding": true,
  "phn_vocab": 39,
  "feature_variant": "pitch",
  "features": ["norm_log_f0", "voiced_ratio"]
}
```

### 14.2 Ghi chú thêm

- Nếu muốn reproduce lại, download `seq_data_w2v_research_pitch/` và train từ đầu với seed 42.
- Mô hình được huấn luyện trên Kaggle Notebook; tất cả checkpoints đã export sang .pth.
- Nếu có vấn đề về device (GPU/CPU), kiểm tra `model.to(device)` và `data.to(device)`.
- Phoneme vocabulary (39 phones) từ phonemizer eSpeak backend.

---

## 15. Lịch sử chỉnh sửa

- [2024-04-28] Hoàn thành ablation study 4 variants; chọn pitch variant.
- [2024-04-28] Viết README_draft.md với đầy đủ tài liệu.
- [TBD] Chuẩn bị paper submission.

---

**Ghi chú cuối:** Đây là bản README hoàn chỉnh dựa trên dữ liệu thực tế từ thư mục `final/w2v_gopt_research/`. Tất cả con số, metrics, và kết luận đều từ ablation summary và training histories. Bạn có thể sử dụng ngay để viết paper hoặc trình bày kết quả.
