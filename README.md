# W2V_GOP Pronunciation API

Tai lieu nay duoc viet lai theo kieu de hieu cho nguoi moi.
Ban khong can biet code van co the test va dung duoc.

## 1) Dich vu nay dung de lam gi?

Dich vu nhan:
- text (cau tieng Anh ban muon nguoi hoc doc)
- audio (file ghi am nguoi hoc)

Dich vu tra ve:
- diem phat am
- nhan muc do (Weak / Fair / Good)
- thong tin tong ket theo cau hoac theo nhieu turn hoi thoai

## 2) Ban can chuan bi gi?

- May da co Python va virtual environment `.venv`
- Thu muc du an o `C:\Users\tienanh\Desktop\GOP_AI`
- Model da nam trong `W2V_GOP/model`

## 3) Chay server trong 1 phut

Mo PowerShell tai thu muc du an, chay lan luot:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r .\W2V_GOP\requirements.txt
cd .\W2V_GOP
uvicorn app.main:app --host 0.0.0.0 --port 5005
```

Khi thay dong bao server da chay, mo trinh duyet:
- http://127.0.0.1:5005/docs

Luu y quan trong:
- Khong mo bang `http://0.0.0.0:5005` (dia chi nay la dia chi bind, khong phai dia chi de truy cap bang browser).
- Neu mo tren cung may, dung `127.0.0.1` hoac `localhost`.

## 4) Co may endpoint?

Public API hien tai co 2 endpoint cham diem:

1. `POST /score`
- Dung khi cham diem 1 cau
- Input: `text` + `audio`

2. `POST /score/conversation/summary`
- Dung khi tong ket nhieu cau (nhieu turn)
- Input: danh sach `texts` + danh sach `audios` (so luong phai bang nhau)

Them endpoint kiem tra tinh trang:
- `GET /health`

## 5) Audio co can tu xu ly truoc khong?

Thong thuong la KHONG can.
Server tu lam cac buoc sau:
- chuyen ve mono 16kHz
- denoise (giam nhieu)
- trim silence
- VAD (giu vung co tieng noi)
- normalize

Ban chi can dam bao:
- giong noi ro
- han che tap am lon
- moi file nen trong khoang 1-8 giay

Neu audio qua ngan hoac qua yeu, server co the tu choi request.

## 6) Cach test nhanh tren Swagger (khong can code)

1. Mo `http://127.0.0.1:5005/docs`
2. Chon endpoint `POST /score`
3. Bam `Try it out`
4. Dien `text` (vi du: `i am a student`)
5. Chon file audio
6. Bam `Execute`
7. Xem ket qua JSON

## 7) Test bang PowerShell (copy va chay)

### 7.1 Cham diem 1 cau

```powershell
$uri = "http://127.0.0.1:5005/score"
$form = @{
  text = "i am a student"
  audio = Get-Item "C:\Users\tienanh\Desktop\GOP_AI\recorded.wav"
}
Invoke-RestMethod -Uri $uri -Method Post -Form $form
```

### 7.2 Tong ket nhieu turn (conversation summary)

```powershell
$uri = "http://127.0.0.1:5005/score/conversation/summary"
$form = @{
  texts_json = "[\"my name is tien anh\",\"i am 22 years old\"]"
  include_turn_details = "true"
  audios = @(
    (Get-Item "C:\Users\tienanh\Desktop\GOP_AI\recorded.wav"),
    (Get-Item "C:\Users\tienanh\Desktop\GOP_AI\recorded.wav")
  )
}
Invoke-RestMethod -Uri $uri -Method Post -Form $form
```

## 8) Doc ket qua nhu the nao?

Cac field de nhin nhanh:
- `overall_score_0_5`: diem tong tren thang 5
- `overall_score_0_100`: diem tong tren thang 100
- `overall_label`: nhan danh gia (Weak / Fair / Good)
- `sentence_band`: muc do theo nhom

Field IPA target cho endpoint `/score` (dung cho lop control):
- `ipa_target_text`: IPA muc tieu cua ca cau, vd `a r | j u | o k eI`
- `ipa_target_words`: IPA theo tung tu
- `ipa_target_tokens`: IPA theo tung token
- `ipa_target_mode`: che do sinh IPA (`phonemizer_espeak` hoac fallback)

Vi du nho:

```json
{
  "text": "are you okay?",
  "ipa_target_text": "a r | j u | o k eI",
  "ipa_target_words": [
    {"index": 1, "word": "are", "ipa": "a r", "phones": ["a", "r"]},
    {"index": 2, "word": "you", "ipa": "j u", "phones": ["j", "u"]},
    {"index": 3, "word": "okay", "ipa": "o k eI", "phones": ["o", "k", "eI"]}
  ]
}
```

Neu goi `conversation/summary`, ban se co them:
- `turn_count`: tong so turn gui len
- `processed_turn_count`: so turn hop le da cham
- `signals`: diem pronunciation va grammar
- `modules`: cac nhom loi/noi dung can cai thien
- `common_issues`: loi lap lai nhieu lan

## 9) Loi thuong gap va cach xu ly

1. Loi text/audio length mismatch
- Nguyen nhan: so cau va so file audio khong bang nhau
- Cach sua: dam bao 1 cau di kem 1 file audio

2. Loi audio file is empty
- Nguyen nhan: upload file rong hoac doc file loi
- Cach sua: kiem tra lai file truoc khi gui

3. Loi audio rejected (too short / low energy)
- Nguyen nhan: audio qua ngan hoac qua nho
- Cach sua: ghi am lai ro hon, dai hon

4. Khong vao duoc trang test API
- Dung URL: `http://127.0.0.1:5005/docs`
- Khong dung: `http://0.0.0.0:5005`

## 10) Goi y dung trong san pham

- Neu app cua ban dang cham tung cau: dung `/score`
- Neu cuoi buoi moi tong ket toan bo: dung `/score/conversation/summary`
- Luu lai cac truong: `overall_score_0_100`, `overall_label`, `sentence_band`, `modules`, `common_issues`

## 11) Ghi chu ky thuat ngan gon (cho team dev)

- Service load model khi startup
- Pipeline phoneme su dung eSpeak (`phonemizer_espeak`)
- Neu eSpeak loi, request se fail (khong co fallback im lang)
- Metadata tra ve co version de de doi soat model va calibration

---

Neu ban muon, co the them mot file `README_non_technical_vi.md` rieng cho business/user, giu `README.md` cho dev. Hien tai file nay dang theo huong de hieu cho nguoi moi.
