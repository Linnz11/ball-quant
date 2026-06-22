import tempfile
import unittest
from pathlib import Path

from ball_quant.adapters.ticai import load_ticai_matches


class TicaiParserTest(unittest.TestCase):
    def test_parse_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sp.csv"
            path.write_text(
                "match_id,date,home,away,spf_home,spf_draw,spf_away,handicap,rq_home,rq_draw,rq_away\n"
                "001,2026-06-14,Netherlands,Japan,1.55,3.90,5.60,-1,2.78,3.55,2.05\n",
                encoding="utf-8",
            )
            matches = load_ticai_matches(str(path))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].home, "Netherlands")
        self.assertEqual(matches[0].handicap, -1)

    def test_parse_html_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sp.html"
            path.write_text(
                "<table><tr><th>编号</th><th>日期</th><th>主队</th><th>客队</th>"
                "<th>主胜</th><th>平</th><th>主负</th><th>让球</th><th>让胜</th><th>让平</th><th>让负</th></tr>"
                "<tr><td>002</td><td>2026-06-14</td><td>Ivory Coast</td><td>Sweden</td>"
                "<td>2.60</td><td>3.10</td><td>2.55</td><td>1</td><td>1.45</td><td>3.80</td><td>5.40</td></tr></table>",
                encoding="utf-8",
            )
            matches = load_ticai_matches(str(path))
        self.assertEqual(matches[0].match_id, "002")
        self.assertEqual(matches[0].rq_away, 5.40)


if __name__ == "__main__":
    unittest.main()
