# 스냅샷: 2026-08-14 구조 전수 감사 이전 결과

`testbed/results/`에 남아있던 Track A 270개(3개 데이터셋 × 90개 조합, Track B
3개 조합은 애초에 실행된 적 없어 파일 없음)를 그대로 보존한 것이다.

**왜 지웠는가**: 이 파일들은 전부 2026-07-31~08-04 사이 생성됐는데, 그 이후
`testbed/data/dataset_loader.py`(class-incremental 분할로 전면 재작성),
`testbed/components/{cade,cndids,ssf}/*.py`(CADE class-aware pairing, SSF
InfoNCE 손실항, SSF 메모리 버퍼 drift 방향, CND-IDS multi-teacher LwF — 4건
구조적 충실도 보강), `testbed/pipeline/cl_client.py`, `testbed/base/models.py`가
전부 수정됐다(2026-08-12). `grid_runner.py`의 결과 캐싱(`if os.path.exists
(out_path): continue`)은 코드 버전을 검사하지 않으므로, 이 파일들을 지우지
않고 그대로 그리드를 재실행하면 위 수정 이전 버전 코드로 계산된 낡은 결과가
"최신"인 것처럼 재사용된다 — 실제로 순정 CADE 조합 BWT가 class-incremental
전환 전후로 +0.013→-0.177까지 바뀌는 것으로 실측 확인됐다(`docs/
metric_justification.md` 참고). `testbed/results/`는 이 스냅샷을 만든 뒤
비웠고, 다음 그리드 실행은 처음부터 다시 계산된다.

참고로 `testbed/results/`는 이미 git에 커밋되어 있었으므로(`git log --
oneline -- testbed/results/`로 확인 가능) git 히스토리로도 복원 가능하다 —
이 폴더는 git 명령 없이 바로 찾아볼 수 있도록 만든 편의용 사본이다.
