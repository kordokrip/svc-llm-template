import re, json
from datetime import datetime
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
from dotenv import load_dotenv

load_dotenv()

# I/O base paths
RAW = Path("data/raw/excel")
OUT = Path("data/processed/md")
OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------

def _s(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

# tolerant multi-column picker
def pick(row, *cols):
    for c in cols:
        try:
            s = row.get(c)
        except Exception:
            s = None
        s = _s(s)
        if s:
            return s
    return ""

# like pick(), but runs through to_date_str on the picked value

def pick_date(row, *cols):
    for c in cols:
        try:
            v = row.get(c)
        except Exception:
            v = None
        s = to_date_str(v)
        if s:
            return s
    return ""

_def_time_re = re.compile(r"^\d{1,2}:\d{2}")
_date_prefix_re = re.compile(r"^\d{4}-\d{2}-\d{2}")


def to_date_str(x):
    """Return YYYY-MM-DD or HH:MM; empty string for NaT/None/blank.
    Robust to Excel NaT, numeric time fractions, and date serials.
    """
    # fast empty checks
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    # pandas / python datetimes
    if isinstance(x, pd.Timestamp):
        if pd.isna(x):
            return ""
        return x.strftime("%Y-%m-%d")
    if isinstance(x, datetime):
        return x.strftime("%Y-%m-%d")

    # Excel numerics (time fractions or date serials)
    if isinstance(x, (int, float)):
        # time fraction (0<=x<1)
        xf = float(x)
        if 0 <= xf < 1:
            total_seconds = int(round(xf * 24 * 3600))
            hh = total_seconds // 3600
            mm = (total_seconds % 3600) // 60
            return f"{hh:02d}:{mm:02d}"
        # plausible Excel serial date range only; otherwise keep as-is
        if 59 < xf < 60000:  # 1899-12-30 origin; ignore outliers like 42100203
            try:
                return pd.to_datetime(xf, unit='D', origin='1899-12-30').strftime('%Y-%m-%d')
            except Exception:
                return str(x)
        return str(x)

    # strings and others
    s = str(x).strip()
    if s in ("NaT", "nan", "NaN", ""):
        return ""
    # HH:MM
    if _def_time_re.match(s):
        return s
    # YYYY-MM-DD ...
    if _date_prefix_re.match(s):
        return s.split(" ")[0]
    # best-effort parse
    try:
        ts = pd.to_datetime(s, errors='coerce')
        if pd.isna(ts):
            return s
        return ts.strftime('%Y-%m-%d')
    except Exception:
        return s


def md_header(meta: dict) -> str:
    return "---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items()) + "\n---\n\n"


def _save(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

# -----------------------------
# A) Warranty: 행 → 카드
# -----------------------------

def warranty_to_md(xlsx: str, sheet: str, out_name: str):
    p = RAW / xlsx
    if not p.exists():
        print(f"[SKIP] {p} not found")
        return
    df = pd.read_excel(p, sheet_name=sheet)
    cards = []
    for i, row in df.fillna("").iterrows():
        # tolerant column mapping
        country = row.get("Country", row.get("국가", ""))
        site = row.get("Location", row.get("Site", row.get("Fab", "")))
        customer = row.get("Customer", row.get("고객사", ""))
        model = row.get("MODEL", row.get("Model", ""))
        anchor = pick(row, "기준 날짜", "Warranty기준", "Warranty 기준", "Unnamed: 11")
        duration = pick(row, "기간", "기간(일)")
        meta = dict(
            doc_id="warranty_rules_v1",
            sheet=sheet,
            row_id=int(i),
            country=_s(country),
            site=_s(site),
            customer=_s(customer),
            model=_s(model),
            warranty_anchor=_s(anchor),
            duration_days=_s(duration),
            synonyms=[["Fab In", "설비반입"], ["Turn-on", "가동개시"], ["FAT", "시운전완료"]],
            updated_at="",
        )
        body = (
            f"### 워런티 규정({_s(site)} / {_s(customer)} / {_s(model)})\n"
            f"- 시작 기준: **{_s(anchor)}**\n"
            f"- 기간: **{_s(duration)}일**\n"
        )
        cards.append(md_header(meta) + body)
    _save(OUT / f"{out_name}.md", "\n".join(cards))
    print(f"[OK] warranty → {OUT/out_name}.md ({len(cards)} cards)")

# -----------------------------
# B) Naming Rule: 버전 중심 요약
# -----------------------------

def naming_to_md(xlsx: str, sheet: str, out_name: str):
    p = RAW / xlsx
    if not p.exists():
        print(f"[SKIP] {p} not found")
        return
    df = pd.read_excel(p, sheet_name=sheet)
    # 유연한 컬럼 탐색
    # 버전열은 두 번째 열(관례) 혹은 '버전' 명칭을 우선 사용
    if "버전" in df.columns:
        ver = df["버전"].astype(str).tolist()
    else:
        ver = df.iloc[:, 1].astype(str).tolist()
    # '최신 버전' 키워드가 포함된 열에서 설명 수집
    desc_cols = [c for c in df.columns if "최신" in str(c) or "버전" in str(c)]
    desc_series = df[desc_cols[0]] if desc_cols else df.iloc[:, 0]
    desc = desc_series.fillna("").astype(str).tolist()

    lines = [
        md_header({"doc_id": "naming_rule_v1", "sheet": sheet, "updated_at": ""})
        + "# Naming Rule — 버전 요약\n| 버전 | 주요 변경 키워드 |\n|---|---|\n"
    ]
    for v, d in zip(ver, desc):
        if v and v != "nan" and d:
            first_line = d.splitlines()[0][:100]
            lines.append(f"| {v} | {first_line}… |")
    lines.append("\n## 버전별 상세\n")
    for v, d in zip(ver, desc):
        if v and v != "nan" and d:
            bullets = "\n".join(f"- {x}" for x in d.splitlines() if x.strip())
            lines.append(f"### v{v}\n{bullets}\n")
    _save(OUT / f"{out_name}.md", "\n".join(lines))
    print(f"[OK] naming → {OUT/out_name}.md")

# -----------------------------
# C) 안전재고: 공식/절차 설명서(정적 템플릿)
# -----------------------------

def safety_formula_md(out_name: str):
    meta = {"doc_id": "safety_stock_formula_v1", "updated_at": ""}
    body = (
        "# 안전재고 산출 기준(요약)\n"
        "- 용어: Fail률, 교체율, Lead Time(미발주 포함) …\n"
        "- 기본 공식 예시: `안전재고 ≈ 최근 Fail률 × 가동수량 × Lead Time(개월)`\n"
        "- 절차:\n"
        "  1) 품번별 월평균 Fail률 산정\n"
        "  2) 최근 Fail률 가중 반영\n"
        "  3) Lead Time 반영\n"
        "- FAQ:\n"
        "  - Q. ‘Turn-on일’과 ‘교체일’ 데이터는 어떻게 반영?  \n"
        "    A. …(사내 기준 기입)\n"
    )
    _save(OUT / f"{out_name}.md", md_header(meta) + body + "\n")
    print(f"[OK] safety formula → {OUT/out_name}.md")

# -----------------------------
# D) BM 업무 (Service/CS) 기록: 행 → 케이스 카드
# -----------------------------

def bm_to_md(xlsx: str, sheet: str, out_name: str):
    p = RAW / xlsx
    if not p.exists():
        print(f"[SKIP] {p} not found")
        return
    df = pd.read_excel(p, sheet_name=sheet)

    # 가벼운 전처리
    df = df.fillna("")
    cards = []
    for i, row in df.iterrows():
        # BM만 필터(없으면 전체 통과)
        work_type = _s(row.get("작업유형", ""))
        if work_type and work_type.upper() != "BM":
            continue

        title = f"{_s(row.get('고객사'))} / {_s(row.get('단지'))}{_s(row.get('라인'))} / {_s(row.get('설비'))} {_s(row.get('챔버'))} ({_s(row.get('Model', row.get('MODEL', '')))} )".strip()

        meta = dict(
            doc_id="bm_cases_v1",
            sheet=sheet,
            row_id=int(i),
            team=_s(row.get("TEAM")),
            unit=_s(row.get("운영단위")),
            customer=_s(row.get("고객사")),
            site=_s(row.get("단지")),
            line=_s(row.get("라인")),
            process=_s(row.get("공정")),
            sub_process=_s(row.get("세부공정")),
            tool=_s(row.get("설비")),
            chamber=_s(row.get("챔버")),
            model=_s(row.get("Model", row.get("MODEL", ""))),
            sn_in=_s(row.get("S/N (In)")),
            sn_out=_s(row.get("S/N(Out)")),
            turn_on=pick_date(row, "Turn-on일자", "Turn-on", "Turn On", "TURN_ON"),
            work_type=work_type,
            symptom=_s(row.get("현상")),
            category=_s(row.get("작업구분")),
            warranty=_s(row.get("Warranty \nIn/Out", row.get("Warranty In/Out", row.get("Warranty", "")))),
            replace_part=_s(row.get("교체파트")),
            wafer_loss=_s(row.get("웨이퍼로스")),
            wrs=_s(row.get("WRS")),
            biz_owner=_s(row.get("현업담당자")),
            biz_contact=_s(row.get("현업연락처")),
            work_date=pick_date(row, "작업일자", "작업일", "WorkDate"),
            start_time=pick_date(row, "작업시작시간", "시작시간", "Start Time", "StartTime"),
            end_date=pick_date(row, "작업종료일자", "종료일자", "End Date", "EndDate"),
            end_time=pick_date(row, "작업종료시간", "종료시간", "End Time", "EndTime"),
            duration_min=_s(row.get("작업시간")),
            updated_at="",
        )

        # 본문(검색 최적화된 계층형 문단)
        body_lines = [
            f"### BM 케이스 — {title}",
            "**요약**",
            f"- 증상: {meta['symptom']}",
            f"- 분류: {meta['category']}  |  워런티: {meta['warranty']}  |  교체파트: {meta['replace_part']}",
            "",
            "**현장 정보**",
            f"- 고객/공정: {meta['customer']} / {meta['process']} → {meta['sub_process']}",
            f"- 설비: {meta['tool']} {meta['chamber']}  |  모델: {meta['model']}  |  S/N(In/Out): {meta['sn_in']} / {meta['sn_out']}",
            f"- Turn-on: {meta['turn_on']}",
            "",
            "**작업 이력**",
            f"- 일시: {meta['work_date']} {meta['start_time']} ~ {meta['end_time']}  (총 {meta['duration_min']}분)",
            f"- 담당자: {meta['biz_owner']}  |  연락처: {meta['biz_contact']}",
        ]
        if meta['wafer_loss']:
            body_lines.append(f"- 웨이퍼로스: {meta['wafer_loss']}")
        if _s(row.get("비고")):
            body_lines.append(f"- 비고: {_s(row.get('비고'))}")

        card = md_header(meta) + "\n".join(body_lines) + "\n"
        cards.append(card)

    _save(OUT / f"{out_name}.md", "\n\n".join(cards))
    print(f"[OK] bm → {OUT/out_name}.md ({len(cards)} cards)")

# -----------------------------
# Main (batch)
# -----------------------------
if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) Warranty
    warranty_to_md("해외 Warranty 기준.xlsx", sheet="Warrant 기준", out_name="warranty_rules")

    # 2) Naming Rule
    naming_to_md("NAMING RULE 기준.xlsx", sheet="Naming Rule", out_name="naming_rule")

    # 3) 안전재고 공식(설명)
    safety_formula_md("safety_stock_formula")

    # 4) BM 업무 기록
    bm_to_md("K2024~25 BM_250926.xlsx", sheet="Sheet1", out_name="bm_cases")

    print("MD written to", OUT)