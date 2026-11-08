"""The multi-product paper run (user 2026-11-08: "后续品种可能增加 ... 不然到时候不好维护"): its review result ids
come after the highest of any policy of the formal run, one per product."""
from dreamerv3.thinking_student.paper_run import next_result


def test_next_result(tmp_path):
  assert next_result(tmp_path) == 'r0001'
  for policy, result in (('pi_0005', 'r0009'), ('pi_0009', 'r0029'), ('pi_0009', 'r0003')):
    (tmp_path / 'review' / 'student' / policy / 'PAPER' / result).mkdir(parents=True)
  (tmp_path / 'review' / 'student' / 'pi_0005' / 'PAPER' / 'r0031.tmp').mkdir()
  assert next_result(tmp_path) == 'r0030'
