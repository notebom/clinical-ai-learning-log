# AI System Literacy Report

> AI 회사에서 일하는 비전공자가 AI 시스템을 이해하기 위해 정리한 기술·통계·수학 문해력 학습 보고서입니다.

AI 제품을 이해하려면 모델이 실행되는 **컴퓨터 시스템**, 모델 성능을 판단하는 **통계적 기준**, 내부 계산을 읽는 **기초 수학**을 함께 알아야 합니다. 이 저장소는 엄밀한 증명이나 구현 튜토리얼보다, 기술 회의에서 구조·숫자·계산 흐름을 올바르게 해석하고 질문하는 능력에 초점을 둡니다.

## Learning Tracks

### 1. Computer Systems Literacy

AI 서비스가 실제로 실행되는 기반을 다룹니다.

- CPU, GPU, RAM, VRAM, Storage
- Operating System, Process, Thread
- Server, Client, HTTP, HTTPS, DNS, TLS
- API, Authentication, Authorization
- Latency, Throughput, Cache, Queue
- Docker, Kubernetes, Cloud, Observability

[Computer Systems Track 바로가기](./docs/01.%20computer-systems/README.md)

### 2. Statistics for AI Literacy

AI 모델의 결과를 해석하고 평가하는 기준을 다룹니다.

- Mean, Variance, Standard Deviation
- Probability and Conditional Probability
- Train/Test Split and Data Leakage
- Confusion Matrix, Sensitivity, Specificity
- PPV, NPV, Prevalence
- AUROC, AUPRC, Calibration
- Regression, p-value, Prediction vs Causation

[Statistics for AI Track 바로가기](./docs/03.%20statistics-for-ai/README.md)

### 3. Math for AI Literacy

AI 모델 내부의 계산 흐름을 읽기 위한 수학적 직관을 다룹니다.

- Function, Parameter, Linear Transformation
- Weight, Bias, Activation Function
- Exponential, Logarithm, Numerical Stability
- Sigmoid, Softmax, Loss, Cross Entropy
- Vector, Embedding, Dot Product, Cosine Similarity
- Matrix, Tensor, Shape, Broadcasting
- Derivative, Gradient, Backpropagation, Optimization

[Math for AI Track 바로가기](./docs/04.%20math-for-ai/README.md)

## Structure

```text
00_basic_for_ai/
├── README.md
├── docs/
│   ├── 01. computer-systems/
│   ├── 02. python-basic/
│   ├── 03. statistics-for-ai/
│   └── 04. math-for-ai/
├── assets/
│   ├── computer-systems/
│   ├── statistics-for-ai/
│   └── math-for-ai/
└── references/
    ├── computer-systems.md
    ├── statistics-for-ai.md
    └── math-for-ai.md
```

## How the Two Tracks Connect

| 질문 | 필요한 문해력 |
|---|---|
| 요청이 왜 느린가? | 서버 경로, 자원 병목, latency, observability |
| 모델 확률 0.8을 믿을 수 있는가? | calibration, prevalence, external validation |
| 운영에서 양성 알림이 너무 많은 이유는? | threshold, PPV, base rate, queue capacity |
| 다른 병원에서 성능이 떨어진 이유는? | data distribution shift, leakage, subgroup validation |
| GPU를 늘리면 문제가 해결되는가? | CPU/GPU/I/O 병목과 실제 모델 평가 목표 |
| logits, embedding, gradient는 무엇인가? | 함수, 벡터, 행렬, 미분과 최적화 |
| loss가 감소하는데 실제 성능은 왜 나쁜가? | 목적함수, 과적합, 독립 평가와 calibration |

## Target Reader

- AI 회사에서 일하지만 컴퓨터공학·통계학 전공자가 아닌 사람
- 의료 AI의 기술 구조와 성능 지표를 함께 이해하려는 기획자·PM·연구 지원자
- 모델 성능 숫자를 실제 제품 및 임상 의사결정과 연결하고 싶은 사람
- AI 모델 설명에서 나오는 수식과 tensor shape을 직관적으로 따라가고 싶은 사람

## Key Takeaway

AI 시스템을 이해한다는 것은 모델 하나를 아는 것이 아닙니다. 모델이 어떤 컴퓨터와 서비스 구조 위에서 실행되는지, 어떤 수학적 계산으로 입력을 변환하는지, 성능 숫자가 어떤 데이터와 평가 조건에서 만들어졌는지 함께 이해하는 일입니다.
