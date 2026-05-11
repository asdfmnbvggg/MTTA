
# MTTA: Multi-bin Test-Time Adaptation for Open-set Wafer Defect Detection

반도체 Wafer Map 결함 데이터에서 **Open-set Test-Time Adaptation(TTA)** 문제를 다룬다.  
기존 TTA 방법은 테스트 데이터가 학습 클래스 안에 포함된다는 close-set 가정을 전제로 하지만, 실제 제조 환경에서는 기존에 학습하지 않은 새로운 결함 유형이 함께 등장할 수 있다. 본 연구는 이러한 상황에서 ID 결함과 OOD 결함을 구분하고, 테스트 시점에서 모델을 안정적으로 적응시키기 위한 MTTA 방법을 실험한다.

---

## 연구 개요

### 문제 정의

실제 반도체 제조 공정에서는 다음 두 가지 분포 변화가 동시에 발생할 수 있다.

- **Covariate Shift**: 기존에 학습한 결함 클래스는 유지되지만, 장비 설정·노이즈·공정 조건 변화로 데이터 형태가 달라지는 상황
- **Semantic Shift**: 학습 시점에 존재하지 않았던 새로운 결함 클래스가 테스트 시점에 등장하는 상황

따라서 모델은 기존 결함 클래스에 대해서는 안정적으로 적응하면서도, 처음 보는 결함 유형은 OOD로 분리할 수 있어야 한다.

### 제안 방법

본 연구의 MTTA는 다음 구성 요소를 기반으로 한다.

1. **ResNet-18 기반 Feature Extraction**
2. **Mahalanobis Distance 기반 ID/OOD Score 계산**
3. **Multi-bin Distribution Modeling**을 통한 score 분포 분할
4. **Selective Weighted Entropy Optimization**을 통한 안정적인 TTA 업데이트

핵심 아이디어는 score 분포를 단순히 두 개의 그룹으로 나누는 대신, 여러 bin으로 나눈 뒤 신뢰도가 높은 극단 구간만 선택적으로 업데이트에 사용하는 것이다.

---

## 데이터셋

본 프로젝트는 **WM-811K Wafer Map Dataset**을 사용한다.

실험에서는 결함이 존재하는 wafer map만 사용하며, 정상 클래스는 제외한다. Open-set 환경 구성을 위해 결함 클래스를 다음과 같이 나눈다.

| 구분 | 클래스 |
|---|---|
| ID / Known | Center, Edge-Loc, Edge-Ring, Loc |
| OOD / Unknown | Donut, Random, Scratch, Near-full |

---

## 프로젝트 구조

```text
MTTA-main/
├── README.md
└── MTTA/
    └── MTTA/
        ├── requirements.txt
        ├── data/
        │   ├── load_data.py
        │   ├── data_delate.py
        │   ├── data_preprocessing.py
        │   ├── data_split_id_ood.py
        │   ├── data_split_train_test.py
        │   └── data_check.py
        └── wafer/
            ├── Resnet_18.py
            ├── load_Resnet_18.py
            ├── data_unknown.py
            ├── main.py
            ├── tent.py
            ├── utils.py
            └── run.sh
````

---

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd MTTA-main/MTTA/MTTA
```

### 2. 가상환경 생성

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

> GPU 환경에서 실행할 경우, 사용 중인 CUDA 버전에 맞는 PyTorch 설치가 필요할 수 있다.

---

## 데이터 준비

### 1. WM-811K 데이터 다운로드

```bash
cd data
python load_data.py
```

### 2. 데이터 전처리

각 전처리 스크립트 안의 `DATA_ROOT` 또는 `pkl_path`를 본인 환경의 데이터 경로로 수정한 뒤 실행한다.

```bash
python data_delate.py
python data_preprocessing.py
python data_split_id_ood.py
python data_split_train_test.py
```

전처리 후 다음과 같은 pkl 파일을 준비한다.

```text
LSWMD_prepro.pkl
LSWD_id.pkl
LSWD_ood.pkl
LSWD_id_train.pkl
LSWD_id_test.pkl
```

---

## 모델 학습

ID 클래스 데이터로 ResNet-18 source model을 학습한다.

```bash
cd ../wafer
python Resnet_18.py \
  --train_pkl ../data/LSWD_id_train.pkl \
  --test_pkl ../data/LSWD_id_test.pkl \
  --save_dir ./output \
  --epochs 30 \
  --batch_size 128
```

학습이 완료되면 다음 checkpoint가 저장된다.

```text
./output/resnet18_wafer_best.pth
```

---

## 평가 및 TTA 실행

### Source model 평가

```bash
python main.py \
  --adaptation source \
  --ckpt_path ./output/resnet18_wafer_best.pth \
  --id_pkl ../data/LSWD_id_test.pkl \
  --ood_pkl ../data/LSWD_ood.pkl \
  --batch_size 256 \
  --save_dir ./output
```

### Tent 기반 TTA 실행

```bash
python main.py \
  --adaptation tent \
  --ckpt_path ./output/resnet18_wafer_best.pth \
  --id_pkl ../data/LSWD_id_test.pkl \
  --ood_pkl ../data/LSWD_ood.pkl \
  --batch_size 256 \
  --steps 1 \
  --lr 1e-3 \
  --criterion ent \
  --alpha 0.5 \
  --save_dir ./output
```

### UniEnt 계열 criterion 사용 예시

```bash
python main.py \
  --adaptation tent \
  --ckpt_path ./output/resnet18_wafer_best.pth \
  --id_pkl ../data/LSWD_id_test.pkl \
  --ood_pkl ../data/LSWD_ood.pkl \
  --batch_size 256 \
  --steps 1 \
  --lr 1e-3 \
  --criterion ent_ind_ood \
  --alpha 0.5 1.0 \
  --save_dir ./output
```

---

## 실험 결과 요약

| Method         |  Total Acc |      AUROC |  FPR@TPR95 |       OSCR |
| -------------- | ---------: | ---------: | ---------: | ---------: |
| Source         |     95.84% |     70.29% |     78.68% |     68.86% |
| Tent           |     92.07% |     30.98% |     97.07% |     27.63% |
| Tent + UniEnt  |     87.08% |     28.87% |     97.94% |     23.81% |
| Tent + UniEnt+ |     90.49% |     35.55% |     95.66% |     31.12% |
| MTTA           | **96.09%** | **75.91%** | **77.34%** | **74.51%** |

MTTA는 Source 모델보다 높은 AUROC과 OSCR을 기록했으며, 기존 entropy minimization 기반 TTA 방법보다 Open-set 환경에서 안정적인 성능을 보였다.

---

## 참고 사항

* `run.sh`는 여러 adaptation 방법을 반복 실행하기 위한 실험용 스크립트이다.
* 현재 `main.py` 기준으로는 `--adaptation source`와 `--adaptation tent`가 실행 가능하다.
* 데이터 전처리 스크립트에는 로컬 Windows 경로가 포함되어 있으므로, 실행 전 본인 환경에 맞게 경로를 수정해야 한다.
* pkl 데이터와 학습된 checkpoint 파일은 용량이 클 수 있으므로 GitHub에는 업로드하지 않고, 별도 저장소 또는 Google Drive 등을 통해 관리하는 것을 권장한다.

---

## GitHub 반영 명령어

```bash
git add README.md
git commit -m "docs: update MTTA project README"
git push origin main
```

