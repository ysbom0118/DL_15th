# 04장 파이토치 심화

## 개요
딥러닝 모델의 성능 향상과 안정적인 학습을 위한 고급 기법들을 학습합니다. 가중치 초기화, 정규화, 데이터 증강 등을 다룹니다.

---

## 예제 목록

### 가중치 초기화
| 예제 | 파일명 | 주요 내용 |
|-----|--------|----------|
| 4.01 | `예제 4.01 텐서 초깃값.ipynb` | zeros, ones, normal 초기화 |
| 4.03 | `예제 4.03 가중치 초기화 함수 (1).ipynb` | Xavier, Kaiming 초기화 |
| 4.04 | `예제 4.04 가중치 초기화 함수 (2).ipynb` | apply를 이용한 초기화 적용 |

### 정규화
| 예제 | 파일명 | 주요 내용 |
|-----|--------|----------|
| 4.02 | `예제 4.02 배치 정규화 수행.ipynb` | BatchNorm1d/2d 사용법 |
| 4.05 | `예제 4.05 L1 정칙화 적용 방식.ipynb` | L1 정규화 (Lasso) |
| 4.06 | `예제 4.06 L2 정칙화 적용 방식.ipynb` | L2 정규화 (Ridge) |
| 4.07 | `예제 4.07 가중치 감쇠 적용 방식.ipynb` | weight_decay 매개변수 |
| 4.08 | `예제 4.08 드롭아웃 적용 방식.ipynb` | nn.Dropout 사용법 |

### 그라디언트 제어
| 예제 | 파일명 | 주요 내용 |
|-----|--------|----------|
| 4.09 | `예제 4.09 그라디언트 클리핑 적용 방식.ipynb` | clip_grad_norm_, clip_grad_value_ |

### 데이터 증강
| 예제 | 파일명 | 주요 내용 |
|-----|--------|----------|
| 4.10-4.15 | `예제 4.10~4.15 텍스트 데이터 증강 실습.ipynb` | nlpaug (삽입, 대체, 삭제, 역번역) |
| 4.16-4.24 | `예제 4.16~4.24 이미지 데이터 증강 실습.ipynb` | transforms, imgaug, Mixup |

---

## 핵심 개념

### 가중치 초기화 전략
| 초기화 방법 | 적합한 활성화 함수 |
|-----------|------------------|
| Xavier (Glorot) | Sigmoid, Tanh |
| Kaiming (He) | ReLU, LeakyReLU |

```python
# Xavier 초기화
nn.init.xavier_uniform_(layer.weight)

# Kaiming 초기화
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
```

### 정규화 기법 비교
| 기법 | 특징 |
|-----|------|
| L1 정규화 | 희소 가중치 (특성 선택) |
| L2 정규화 | 작은 가중치 유지 |
| Dropout | 학습 시 노드 무작위 제거 |
| Batch Norm | 레이어 입력 정규화 |

### 데이터 증강
- **텍스트**: 동의어 대체, 역번역, 랜덤 삽입/삭제
- **이미지**: 회전, 뒤집기, 색상 변환, Mixup

---

## 필요 라이브러리
```python
# 기본
import torch
from torch import nn

# 텍스트 증강
import nlpaug.augmenter.word as naw

# 이미지 증강
from torchvision import transforms
import imgaug.augmenters as iaa
```
