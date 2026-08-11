# 스냅샷: 2026-08-11 하이퍼파라미터 감사 이전 결과

이 폴더는 `docs/metric_justification.md`의 "2026-08-11" 날짜 항목들
(공유 백본 hidden_dim/latent_dim SSF 공식 전환, CND-IDS K-means 원 논문
리스트 복귀, CADE 미니배치 학습 도입)을 적용해 전체 그리드를 재실행하기
**직전**의 결과를 보존한 것이다.

- `results/` — 당시 `testbed/results/`의 사본(Track A 270개, Track B 0개 —
  Track B는 이 시점에 label_budget 수정 반영을 위해 이미 삭제된 상태였음).
- `reports/` — 당시 `testbed/reports/`의 사본(리더보드/요약 csv·json).
- `leaderboard_snapshot.html` — 이 시점 데이터로 만든 대화형 대시보드
  (Claude Artifact로 발행됐던 것의 스냅샷).

**주의**: 이 스냅샷의 Track A 결과는 공유 백본 hidden_dim/latent_dim이
NSL-KDD 기준 128/64(SSF 공식대로면 64/32여야 함)로 고정되어 있던 시점의
값이고, `dd=cade` 조합은 CADE 사설 encoder가 사실상 5회 그래디언트
업데이트만 받은 상태였다 — 즉 이 결과들은 "구버전 코드" 기준이며,
2026-08-11 수정 이후의 새 그리드 결과와 직접 비교할 때는 이 배경을
감안해야 한다.

참고로 `testbed/results/`·`testbed/reports/`는 이 스냅샷을 만든 시점 기준
git에 이미 커밋되어 있었으므로(`git log -- testbed/results/`로 확인 가능),
git 히스토리로도 언제든 복원 가능하다 — 이 폴더는 git 명령 없이 바로
찾아볼 수 있도록 만든 편의용 사본이다.
