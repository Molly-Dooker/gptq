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
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone() # 예: (4, 6)
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t() # Conv1D의 경우, W.shape이 (in_features, out_features)가 되도록 전치. 여기서는 Linear로 가정.
                      # Linear의 경우 W.shape은 (out_features, in_features)이므로, GPTQ 내부에서는 (in_features, out_features)로 사용하기 위해 전치.
                      # 예: W.shape -> (6, 4)
        self.rows = W.shape[0] # 예: 6
        self.columns = W.shape[1] # 예: 4
        self.H = torch.zeros((self.columns, self.columns), device=self.dev) # 예: (4, 4)
        self.nsamples = 0

    def add_batch(self, inp, out): # pylint: disable=unused-argument
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3: # inp: (batch_size, seq_len, in_features)
                inp = inp.reshape((-1, inp.shape[-1])) # inp: (batch_size * seq_len, in_features)
            inp = inp.t() # inp: (in_features, batch_size * seq_len) -> (d_in, N). H는 X X^T (N x d_out)(d_out x N) 이므로 X는 (N x d_out) 형태여야 함.
                          # 코드에서는 H가 (d_out, d_out) 이고, X는 보정 샘플 X_i (m x d_out)
                          # 여기서 inp는 보정 샘플의 활성화 값 X를 의미하며, H = sum(X_i^T X_i) 형태 (d_out x d_out)
                          # 레이어 입력 X는 (N, d_in) 이고, 가중치 W는 (d_in, d_out) -> 출력 Y (N, d_out)
                          # 헤시안은 (X^T X) 형태여야 함. (d_in, d_in)
                          # 이 코드에서는 레이어 가중치의 열(column)을 기준으로 양자화하므로, 헤시안은 (d_out, d_out)이 되어야 함.
                          # 따라서, H를 계산할 때 사용되는 'inp'는 실제로는 (보정 데이터 샘플 수, 레이어 출력 차원 수) 또는 이와 유사한 형태여야 함.
                          # 현재 코드의 inp는 레이어 '입력'을 의미하므로, (N, d_in)
                          # H를 (self.columns, self.columns) 즉 (d_out, d_out)로 정의했으므로, inp는 X (샘플 수, d_out) 형태를 가져야 함.
                          # 하지만 Linear 계층의 입력은 (N, d_in)임.
                          # 논문에서는 H_F = 2 X_F X_F^T 로, X_F가 레이어 '입력'임. 따라서 H_F는 (d_in, d_in)
                          # 코드에서는 W를 (d_in, d_out)으로 다루고, 열(d_out)을 양자화. 이 경우 H는 (d_out, d_out)가 맞음.
                          # 이때 X는 (샘플 수, d_out) 형태가 되어야 함. 하지만 `add_batch`의 `inp`는 레이어 입력이므로 (샘플 수, d_in)
                          # 이는 `inp.matmul(inp.t())`가 (d_in, d_in)을 만들게 됨.
                          # 하지만 self.H는 (self.columns, self.columns) = (d_out, d_out)로 초기화됨.
                          # GPTQ 논문에서는 각 행 w (1 x d_in)를 양자화. 이때 H는 (d_in, d_in).
                          # 코드에서는 W를 (d_out, d_in)으로 보고, 각 열 w (d_out x 1)을 양자화 (실제로는 (d_in, d_out)에서 열을 선택).
                          # 이 경우 H는 (d_in, d_in)이 되어야 하고, X는 (샘플 수, d_in)이 됨.
                          # self.columns = W.shape[1] (즉, d_out)로 설정되어 있으므로, H는 (d_out, d_out)
                          # 이는 코드에서 W를 (d_rows, d_columns) = (d_in, d_out)으로 간주하고
                          # 각 *컬럼* $W_{:,j}$ ($d_{in} \times 1$ 벡터)을 순차적으로 양자화하는 것이 아니라,
                          # W를 (d_out, d_in)으로 보고, 각 *컬럼* (실제로는 $W^T$의 컬럼, 즉 $W$의 행)을 양자화 하는 OBS 방식과 유사.
                          # 하지만 `self.H`의 크기가 `(self.columns, self.columns)`이므로, `self.columns`는 양자화 대상이 되는 웨이트 벡터의 차원이 아니라,
                          # 그 웨이트 벡터에 곱해지는 입력 벡터의 차원이어야 함.
                          # 다시 정리: W (d_out, d_in). 양자화는 각 행 w_r (1, d_in)에 대해 수행. H (d_in, d_in).
                          # 코드에서는 W (d_rows=d_in, d_cols=d_out). H (d_cols, d_cols) = (d_out, d_out).
                          # 이는 각 컬럼 W_{:, c} (d_in x 1)을 양자화 하는 것이 아니라,
                          # W의 각 행 $W_{r,:}$ ($1 \times d_{out}$)을 양자화하는 것처럼 보이지만,
                          # H의 차원이 $(d_{out}, d_{out})$이므로, 입력 $X$가 $(N, d_{out})$ 여야 하고, $W$가 $(d_{out}, d_{in})$ 형태여야 함.
                          # 코드에서 `W = layer.weight.data.clone()` (d_out, d_in) 이고, `if transformers.Conv1D: W = W.t()` (d_in, d_out)
                          # Linear의 경우 `W`는 (d_out, d_in)이고, `self.rows = d_out`, `self.columns = d_in`. H는 (d_in, d_in).
                          # 이 설명에서는 Linear 계층이고 W.t()가 없다고 가정, 즉 W.shape (d_out, d_in), self.rows=d_out, self.columns=d_in.
                          # H.shape = (d_in, d_in). 이 경우 inp.shape은 (N, d_in)이고 inp.t()는 (d_in, N). matmul은 (d_in, d_in)
                          # 가정한 shape: W(6,4) -> d_in=6, d_out=4.
                          # 코드의 W는 (rows, columns) = (d_in, d_out) = (6,4). H는 (columns, columns) = (4,4).
                          # inp는 (N, d_in) -> (N, 6). inp.t()는 (6, N). inp.matmul(inp.t())는 (6,6). H와 차원 불일치.

                          # 혼란을 피하기 위해, GPTQ 페이퍼의 정의를 따름: W는 (d_out, d_col) 즉 (d_row, d_col)
                          # 각 행 w (1 x d_col)을 양자화. 헤시안 H (d_col, d_col). 입력 X (N, d_col).
                          # 코드에서는 layer.weight (d_out, d_in).
                          # W = layer.weight.data.clone() -> (d_out, d_in)
                          # self.rows = d_out
                          # self.columns = d_in
                          # H = torch.zeros((self.columns, self.columns)) -> (d_in, d_in)
                          # inp (레이어 입력)은 (N, d_in). inp.t()는 (d_in, N).
                          # H += inp.matmul(inp.t())는 (d_in, d_in). 이것이 논문의 H와 일치.
                          # 가정한 shape: layer.weight (4,6). W (4,6). rows=4, cols=6. H (6,6).
                          # inp (N, 6). inp.t() (6,N). H += (6,N)x(N,6) -> (6,6).
                          # 이 가정으로 주석을 작성.

        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float() # inp: (d_in, N) for Linear. H.shape: (d_in, d_in)
        self.H += inp.matmul(inp.t()) # H: (d_in, d_in)

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone() # W.shape: (d_out, d_in) 예: (4, 6)
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D): # transformers.Conv1D 가중치는 (in_features, out_features)
            W = W.t() # W.shape: (out_features, in_features) 예: (4, 6) -> 논문과 일치시키기 위함
        W = W.float() # W.shape: (d_out, d_in) 예: (4, 6)

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True) # W (4,6)에 대해 scale, zero 계산

        H = self.H # H.shape: (d_in, d_in) 예: (6, 6)
        del self.H
        dead = torch.diag(H) == 0 # dead.shape: (d_in) 예: (6)
        H[dead, dead] = 1
        W[:, dead] = 0 # W의 해당 '열'을 0으로. W.shape (4,6)이므로 6개 열 중 일부.

        if static_groups:
            import copy
            groups = []
             # self.columns는 d_in (예: 6). groupsize (예: 2)
            for i in range(0, self.columns, groupsize): # i = 0, 2, 4
                quantizer = copy.deepcopy(self.quantizer)
                # W[:, i:(i + groupsize)] -> W의 모든 행, i부터 i+groupsize-1까지의 열
                # W.shape (4,6). W[:, 0:2], W[:, 2:4], W[:, 4:6]
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True) # perm.shape: (d_in) 예: (6)
            W = W[:, perm] # W.shape: (d_out, d_in) 예: (4, 6) 열 순서 변경
            H = H[perm][:, perm] # H.shape: (d_in, d_in) 예: (6, 6) 행/열 순서 변경
            invperm = torch.argsort(perm) # invperm.shape: (d_in) 예: (6)

        Losses = torch.zeros_like(W) # Losses.shape: (d_out, d_in) 예: (4, 6)
        Q = torch.zeros_like(W) # Q.shape: (d_out, d_in) 예: (4, 6)

        damp = percdamp * torch.mean(torch.diag(H)) # 스칼라값
        diag = torch.arange(self.columns, device=self.dev) # diag.shape: (d_in) 예: (6)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H) # H.shape: (d_in, d_in) 예: (6, 6) (하삼각)
        H = torch.cholesky_inverse(H) # H.shape: (d_in, d_in) 예: (6, 6) (H^-1)
        H = torch.linalg.cholesky(H, upper=True) # H.shape: (d_in, d_in) 예: (6, 6) (H^-1의 상삼각 콜레스키 인수 U)
        Hinv = H # Hinv.shape: (d_in, d_in) 예: (6, 6)

        # self.columns는 d_in (예: 6). blocksize (예: 2)
        for i1 in range(0, self.columns, blocksize): # i1 = 0, 2, 4
            i2 = min(i1 + blocksize, self.columns) # i2 = 2, 4, 6
            count = i2 - i1 # count = 2

            W1 = W[:, i1:i2].clone() # W1.shape: (d_out, count) 예: (4, 2)
            Q1 = torch.zeros_like(W1) # Q1.shape: (d_out, count) 예: (4, 2)
            Err1 = torch.zeros_like(W1) # Err1.shape: (d_out, count) 예: (4, 2)
            Losses1 = torch.zeros_like(W1) # Losses1.shape: (d_out, count) 예: (4, 2)
            Hinv1 = Hinv[i1:i2, i1:i2] # Hinv1.shape: (count, count) 예: (2, 2)

            for i in range(count): # i = 0, 1
                w = W1[:, i] # w.shape: (d_out) 예: (4)
                d = Hinv1[i, i] # 스칼라값

                if groupsize != -1: # groupsize 예: 2
                    if not static_groups:
                        # (i1 + i) 는 현재 처리중인 W의 실제 열 인덱스
                        if (i1 + i) % groupsize == 0: # (0+0)%2==0, (0+1)%2!=0, (2+0)%2==0, (2+1)%2!=0 ...
                            # W[:, (i1 + i):(i1 + i + groupsize)] -> W의 (i1+i)부터 (i1+i+groupsize-1)까지 열들
                            # W.shape (4,6). 예: W[:, 0:2], W[:, 2:4], W[:, 4:6]
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i # 실제 열 인덱스
                        if actorder: # actorder 사용 시 perm 적용된 인덱스
                            idx = perm[idx].item() # idx는 스칼라
                        self.quantizer = groups[idx // groupsize] # groups 리스트에서 해당 quantizer 가져옴

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten() # w.unsqueeze(1).shape: (d_out, 1) 예: (4,1). q.shape: (d_out) 예: (4)
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d # err1.shape: (d_out) 예: (4)
                # W1[:, i:].shape: (d_out, count-i) 예: i=0 -> (4,2), i=1 -> (4,1)
                # err1.unsqueeze(1).shape: (d_out, 1) 예: (4,1)
                # Hinv1[i, i:].unsqueeze(0).shape: (1, count-i) 예: i=0 -> (1,2), i=1 -> (1,1)
                # matmul 결과: (d_out, count-i) 예: (4,2) 또는 (4,1)
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1 # Q의 (:, i1:i2) 부분에 Q1 할당
            Losses[:, i1:i2] = Losses1 / 2

            # W[:, i2:].shape: (d_out, d_in - i2) 예: i1=0,i2=2 -> (4, 4); i1=2,i2=4 -> (4,2); i1=4,i2=6 -> (4,0)
            # Err1.shape: (d_out, count) 예: (4,2)
            # Hinv[i1:i2, i2:].shape: (count, d_in - i2) 예: i1=0,i2=2 -> (2,4); i1=2,i2=4 -> (2,2); i1=4,i2=6 -> (2,0)
            # matmul 결과: (d_out, d_in - i2)
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])


        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm] # Q.shape: (d_out, d_in) 예: (4, 6)

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t() # Q.shape: (d_in, d_out) 예: (6, 4) -> 원래 Conv1D 가중치 형태로
        # self.layer.weight.shape 예: Linear는 (d_out, d_in) 즉 (4,6). Conv1D는 (d_in, d_out) 즉 (6,4)
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None # __init__에서 할당되므로, 여기서 None으로 재할당하여 명시적 해제
        # self.Losses, self.Trace 등은 fasterquant 내 지역 변수이거나 이 클래스 멤버가 아님
        torch.cuda.empty_cache()