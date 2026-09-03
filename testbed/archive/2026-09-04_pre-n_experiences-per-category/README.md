# 스냅샷: 2026-09-04 n_experiences 정책 변경 이전 결과

`testbed/results/`에 남아있던 NSL-KDD 96개 + UNSW-NB15 96개(총 192개, 두
데이터셋 다 96개 유효 조합 전체 실행 완료 상태)를 그대로 보존한 것이다.
Track B는 라운드 5개 기준(공격 유형과 무관한 고정값)으로, Track A도 마찬가지로
전 데이터셋 n_experiences=5로 실행된 결과다.

**왜 지웠는가**: 커밋 `70de0a2`("n_experiences를 데이터셋 공통 고정값 대신
실제 공격 유형 수로 자동 결정")로 `data/dataset_loader.py`가 라운드 수를
데이터셋 공통 고정값(5) 대신 그 데이터셋 자신의 실제 공격 category 수로
자동 결정하도록 바뀌었다 — NSL-KDD는 4라운드(DoS/Probe/R2L/U2R 각각 1개),
UNSW-NB15는 9라운드(Analysis/Backdoor/DoS/Exploits/Fuzzers/Generic/
Reconnaissance/Shellcode/Worms 각각 1개, 둘 다 실제 데이터로 검증 완료),
CICIDS2018은 14라운드로 예상(실제 검증은 그리드 실행 시 확인 예정)로 라운드
수·라운드 구성 자체가 달라졌다. `grid_runner.py`의 결과 캐싱은 `code_version`
해시로 낡은 결과를 걸러내므로 이 커밋 이후 그리드를 재실행하면 어차피
전부 재계산되지만, `testbed/results/`에 남아있던 옛 파일들이 새 파일과
섞여 있으면 어느 게 새 정책 결과인지 파일명만으로 구분이 안 되고
`perf_matrix`(R행렬) 크기(T×T)도 라운드 수가 달라져 옛 파일은 이제 4×4/9×9가
아니라 5×5라 형식적으로도 다르다 — 혼선을 막기 위해 옮겨두고
`testbed/results/`는 비웠다.

참고로 `testbed/results/`는 이미 git에 커밋되어 있었으므로(`git log --
oneline -- testbed/results/`로 확인 가능) git 히스토리로도 복원 가능하다 —
이 폴더는 git 명령 없이 바로 찾아볼 수 있도록 만든 편의용 사본이다.
