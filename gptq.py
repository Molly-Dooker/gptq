import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer # 양자화를 수행할 레이어
        self.dev = self.layer.weight.device # 레이어 가중치의 디바이스 (CPU 또는 CUDA)
        W = layer.weight.data.clone() # 레이어 가중치 복사
        if isinstance(self.layer, nn.Conv2d): # 2D 합성곱 레이어인 경우
            W = W.flatten(1) # 가중치를 2D로 펼침
        if isinstance(self.layer, transformers.Conv1D): # 1D 합성곱 레이어인 경우 (transformers 라이브러리)
            W = W.t() # 가중치 전치
        self.rows = W.shape[0] # 가중치 행렬의 행 수
        self.columns = W.shape[1] # 가중치 행렬의 열 수
        self.H = torch.zeros((self.columns, self.columns), device=self.dev) # 헤시안 행렬 H 초기화 (열 x 열 크기)
        self.nsamples = 0 # 보정 데이터 샘플 수 초기화

    def add_batch(self, inp, out): # pylint: disable=unused-argument
        # 이 메소드는 헤시안 행렬 H를 계산하기 위해 입력(inp) 배치를 추가합니다.
        # 논문에서는 H = 2 * X * X^T 로 정의되며, 여기서 X는 입력 데이터입니다. [cite: 759]
        # 코드에서는 H를 점진적으로 업데이트하며 평균을 냅니다.
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2: # 입력이 2D 텐서인 경우 (보통 배치 크기 x 특성 수)
            inp = inp.unsqueeze(0) # 배치 차원 추가 (1 x 배치 크기 x 특성 수)
        tmp = inp.shape[0] # 현재 배치의 샘플 수
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            # 선형 계층 또는 Conv1D 계층의 경우
            if len(inp.shape) == 3: # 입력이 3D 텐서인 경우 (샘플 수 x 시퀀스 길이 x 특성 수)
                inp = inp.reshape((-1, inp.shape[-1])) # 2D로 변형 (샘플 수 * 시퀀스 길이 x 특성 수)
            inp = inp.t() # (특성 수 x 샘플 수 * 시퀀스 길이)로 전치
        if isinstance(self.layer, nn.Conv2d): # 2D 합성곱 계층의 경우
            # 입력 데이터를 im2col과 유사한 방식으로 펼쳐서 선형 계층처럼 처리할 수 있도록 함
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp) # (배치 크기 x (채널 수 * 커널 높이 * 커널 너비) x L) 여기서 L은 결과 슬라이딩 윈도우의 수
            inp = inp.permute([1, 0, 2]) # ((채널 수 * 커널 높이 * 커널 너비) x 배치 크기 x L)
            inp = inp.flatten(1) # ((채널 수 * 커널 높이 * 커널 너비) x (배치 크기 * L))
        self.H *= self.nsamples / (self.nsamples + tmp) # 기존 H값에 이전 샘플 수에 대한 가중치 적용 (평균 업데이트)
        self.nsamples += tmp # 총 샘플 수 업데이트
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float() # 입력 데이터 스케일링 (논문의 H = 2XX^T 에 맞추기 위함) [cite: 759]
        # self.H += 2 / self.nsamples * inp.matmul(inp.t()) # H에 현재 배치 기여분 추가 (원래 OBS 공식에 더 가까움)
        self.H += inp.matmul(inp.t()) # 스케일링된 입력을 사용하여 H 업데이트

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        # 이 메소드가 논문의 Algorithm 1에 해당하는 핵심 양자화 로직을 수행합니다. [cite: 802]
        W = self.layer.weight.data.clone() # 양자화할 가중치 W 복사
        if isinstance(self.layer, nn.Conv2d): # 2D 합성곱 레이어 처리
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D): # 1D 합성곱 레이어 처리 (transformers)
            W = W.t()
        W = W.float() # 가중치를 float32로 변환하여 정밀도 높은 계산 수행

        tick = time.time() # 시간 측정 시작

        if not self.quantizer.ready(): # 양자화기(Quantizer)가 아직 설정되지 않았다면
            self.quantizer.find_params(W, weight=True) # W 전체에 대해 양자화 파라미터(scale, zero-point) 계산 [cite: 754]

        # H는 add_batch를 통해 누적된 헤시안 근사 행렬입니다.
        H = self.H # add_batch에서 계산된 헤시안 행렬
        del self.H # 메모리 절약을 위해 원본 H 삭제
        dead = torch.diag(H) == 0 # 헤시안 대각 성분이 0인 열 (거의 업데이트되지 않는 가중치) 식별
        H[dead, dead] = 1 # 해당 열의 헤시안 값을 1로 설정하여 수치적 안정성 확보 (역행렬 계산 시 문제 방지)
        W[:, dead] = 0 # 해당 열의 가중치를 0으로 설정 (영향 없는 가중치)

        # 정적 그룹 (static_groups) 사용 시, 그룹별로 양자화 파라미터를 미리 계산합니다.
        # 논문에서는 명시적으로 언급되지 않았지만, 그룹 양자화의 한 변형입니다. [cite: 884]
        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize): # groupsize 단위로 열 분할
                quantizer = copy.deepcopy(self.quantizer) # 기본 양자화기 복사
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True) # 각 그룹에 대해 양자화 파라미터 계산
                groups.append(quantizer)

        # 활성화 순서 (actorder) 적용: 헤시안 대각 성분(중요도 추정치) 기준으로 열 재정렬
        # 논문에서는 언급되지 않은 휴리스틱으로, LLaMa 모델에서 성능 향상을 보임
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True) # 헤시안 대각값을 기준으로 내림차순 정렬 인덱스
            W = W[:, perm] # 가중치 열 재정렬
            H = H[perm][:, perm] # 헤시안 행/열 재정렬
            invperm = torch.argsort(perm) # 원래 순서로 되돌리기 위한 역순열

        Losses = torch.zeros_like(W) # 양자화로 인한 손실 저장할 텐서
        Q = torch.zeros_like(W) # 양자화된 가중치 저장할 텐서. Algorithm 1의 Q와 동일 [cite: 802]

        # 헤시안 행렬 정규화 (Dampening) 및 콜레스키 분해를 이용한 역행렬 계산 준비
        # 논문 Step 3: Cholesky Reformulation에 해당 [cite: 792, 796, 797]
        # Algorithm 1의 H^-1 <- (2XX^T + lambda*I)^-1 에 해당하며, lambda*I가 dampening 역할 [cite: 802]
        damp = percdamp * torch.mean(torch.diag(H)) # 평균 대각값의 일정 비율로 감쇠값 설정 [cite: 792]
        diag = torch.arange(self.columns, device=self.dev) # 대각선 인덱스
        H[diag, diag] += damp # 헤시안 대각선에 감쇠값 추가 (수치 안정성 및 정칙화 효과)
        H = torch.linalg.cholesky(H) # 콜레스키 분해 L (H = L * L^T)
        H = torch.cholesky_inverse(H) # 콜레스키 분해를 이용하여 H의 역행렬 계산 (H^-1)
        H = torch.linalg.cholesky(H, upper=True) # H^-1의 콜레스키 분해 U (H^-1 = U^T * U), 여기서 U가 Hinv. Algorithm 1의 H^-1 <- Cholesky(H^-1)^T 에 해당 [cite: 802]
        Hinv = H # Hinv는 H^-1의 콜레스키 분해의 상삼각 행렬 U를 의미

        # 열을 블록 단위로 처리 (Lazy Batch-Updates)
        # 논문 Step 2: Lazy Batch-Updates 에 해당하며, B=128을 사용 [cite: 785]
        # Algorithm 1의 바깥쪽 루프 (for i = 0, B, 2B, ... d_col) [cite: 802]
        for i1 in range(0, self.columns, blocksize): # blocksize 단위로 열 처리
            i2 = min(i1 + blocksize, self.columns) # 현재 블록의 끝 인덱스
            count = i2 - i1 # 현재 블록의 열 개수

            W1 = W[:, i1:i2].clone() # 현재 블록의 가중치 복사
            Q1 = torch.zeros_like(W1) # 현재 블록의 양자화된 가중치
            Err1 = torch.zeros_like(W1) # 현재 블록의 양자화 오차 (스케일링됨)
            Losses1 = torch.zeros_like(W1) # 현재 블록의 손실
            Hinv1 = Hinv[i1:i2, i1:i2] # 현재 블록에 해당하는 Hinv의 부분 행렬 (B x B 크기)

            # 블록 내 각 열을 순차적으로 양자화
            # Algorithm 1의 안쪽 루프 (for j = i, ..., i+B-1) [cite: 802]
            for i in range(count): # 현재 블록 내 각 열에 대해 반복
                w = W1[:, i] # 현재 양자화할 열 (벡터)
                d = Hinv1[i, i] # Hinv1의 대각 성분. 논문 Algorithm 1의 [H^-1]_jj 에 해당 [cite: 802]

                # 그룹 양자화 처리
                if groupsize != -1: # 그룹 크기가 지정된 경우
                    if not static_groups: # 동적 그룹: groupsize 마다 양자화 파라미터 재계산
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True) [cite: 861]
                    else: # 정적 그룹: 미리 계산된 그룹별 양자화기 사용
                        idx = i1 + i
                        if actorder: # actorder 사용 시 원래 인덱스로 변환
                            idx = perm[idx].item() # .item() 추가하여 스칼라 값으로 변환
                        self.quantizer = groups[idx // groupsize]

                # 가중치 양자화: w를 quantizer의 scale, zero-point를 사용해 양자화 그리드의 가장 가까운 값으로 반올림
                # 논문 Algorithm 1의 Q_:,j <- quant(W_:,j) 에 해당 [cite: 802]
                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q # 양자화된 가중치 저장
                Losses1[:, i] = (w - q) ** 2 / d ** 2 # 양자화 오차로 인한 손실 계산 (OBQ 수식 (2)의 분자 부분과 유사) [cite: 759]

                # 양자화 오차 계산 및 나머지 가중치 업데이트 (블록 내)
                # 논문 Algorithm 1의 E_:,j-i <- (W_:,j - Q_:,j) / [H^-1]_jj 와 W_:,j:(i+B) <- W_:,j:(i+B) - E_:,j-i * H_j,j:(i+B)^-1 에 해당 [cite: 802]
                err1 = (w - q) / d # 스케일링된 양자화 오차. Algorithm 1의 E_:,j-i [cite: 802]
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) # 블록 내 나머지 가중치 업데이트 (OBS 스타일) [cite: 759]
                Err1[:, i] = err1 # 스케일링된 오차 저장

            Q[:, i1:i2] = Q1 # 전체 Q 행렬에 현재 블록의 양자화된 가중치 저장
            Losses[:, i1:i2] = Losses1 / 2 # 손실 저장 (2로 나누는 것은 OBS 공식과 관련될 수 있음)

            # 블록 외부의 나머지 가중치 업데이트 (전역 업데이트)
            # 논문 Algorithm 1의 W_:, (i+B): <- W_:, (i+B): - E * H_i:(i+B), (i+B):^-1 에 해당 [cite: 802]
            # 논문 수식 (4)와 관련됨: 𝛿_F = -(w_Q - quant(w_Q)) * ([H_F^-1]_QQ)^-1 * (H_F^-1)_:,Q [cite: 786]
            # 여기서 Err1이 (w_Q - quant(w_Q)) / d 에 해당하고, Hinv가 H_F^-1의 콜레스키 인자이므로,
            # Err1.matmul(Hinv[i1:i2, i2:])는 나머지 가중치에 대한 보상 업데이트를 수행합니다.
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG: # 디버깅 목적으로 중간 결과 확인
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize() # CUDA 연산 완료 대기
        print('time %.2f' % (time.time() - tick)) # 총 양자화 시간 출력
        print('error', torch.sum(Losses).item()) # 최종 양자화 오차 합계 출력

        if actorder: # actorder가 적용된 경우, Q를 원래 순서로 복원
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D): # Conv1D의 경우 다시 전치
            Q = Q.t()
        # 최종 양자화된 가중치 Q를 원래 레이어의 가중치로 할당
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        # 메모리 해제
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        # self.H는 fasterquant 시작 시 del H로 이미 해제됨
        # self.Losses
        # self.Trace
        torch.cuda.empty_cache()