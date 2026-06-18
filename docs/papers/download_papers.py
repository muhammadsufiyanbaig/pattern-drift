"""
Download all research papers cited in the pattern-drift system report.
Uses curl (available on Windows 10/11+) for all HTTP operations.
"""
import os
import subprocess
import sys
import time

PAPERS_DIR = os.path.dirname(os.path.abspath(__file__))

PAPERS = [
    {
        "ref": "[1]",
        "filename": "01_gama_2014_survey_concept_drift.pdf",
        "title": "A Survey on Concept Drift Adaptation",
        "authors": "Gama et al. (2014)",
        "urls": [
            "https://kd.cs.uni-kassel.de/wp-content/uploads/2019/07/concept_drift_acmsurvey2014.pdf",
            "https://www.researchgate.net/profile/Joao-Gama/publication/261961254_A_Survey_on_Concept_Drift_Adaptation/links/53ee543b0cf2dc24b3cea87a/A-Survey-on-Concept-Drift-Adaptation.pdf",
        ],
    },
    {
        "ref": "[2]",
        "filename": "02_lu_2018_learning_under_concept_drift_review.pdf",
        "title": "Learning Under Concept Drift: A Review",
        "authors": "Lu et al. (2018)",
        "urls": [
            "https://arxiv.org/pdf/1904.05862.pdf",
        ],
    },
    {
        "ref": "[7]",
        "filename": "—",
        "title": "Continuous Inspection Schemes",
        "authors": "Page (1954)",
        "urls": [],
        "note": "Paywalled — Biometrika/JSTOR",
    },
    {
        "ref": "[8]",
        "filename": "08_bifet_gavaldà_2007_adwin.pdf",
        "title": "Learning from Time-Changing Data with Adaptive Windowing (ADWIN)",
        "authors": "Bifet & Gavaldà (2007)",
        "urls": [
            "https://www.cs.upc.edu/~gabr/papers/sdm07_adwin.pdf",
            "https://epubs.siam.org/doi/pdf/10.1137/1.9781611972771.42",
        ],
    },
    {
        "ref": "[9]",
        "filename": "09_grulich_2018_parallel_adwin.pdf",
        "title": "Scalable Detection of Concept Drifts with Parallel Adaptive Windowing",
        "authors": "Grulich et al. (2018)",
        "urls": [
            "https://www.dfki.de/fileadmin/user_upload/import/9720_grulich-Scalable-Detection-of-Concept-Drifts-on-Data-Streams-with-Parallel-Adaptive-Windowing.pdf",
        ],
    },
    {
        "ref": "[10]",
        "filename": "10_losing_2022_adwin_plus_plus.pdf",
        "title": "Optimizing ADWIN for Steady Streams",
        "authors": "Losing, Hammer & Wersing (2022)",
        "urls": [
            "https://dl.acm.org/doi/pdf/10.1145/3477314.3507074",
        ],
    },
    {
        "ref": "[11]",
        "filename": "11_raab_2020_kswin.pdf",
        "title": "Reactive Soft Prototype Computing for Concept Drift Streams (KSWIN)",
        "authors": "Raab, Heusinger & Schleif (2020)",
        "urls": [
            "https://arxiv.org/pdf/2007.05432.pdf",
        ],
    },
    {
        "ref": "[12]",
        "filename": "—",
        "title": "The Kolmogorov-Smirnov Test for Goodness of Fit",
        "authors": "Massey (1951)",
        "urls": [],
        "note": "Paywalled — JASA/Taylor & Francis",
    },
    {
        "ref": "[13]",
        "filename": "13_gama_2004_ddm.pdf",
        "title": "Learning with Drift Detection (DDM)",
        "authors": "Gama et al. (2004)",
        "urls": [
            "https://link.springer.com/content/pdf/10.1007/978-3-540-28645-5_29.pdf",
            "https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=3cb54f04765c19e7e0580196c29c64e49f63a744",
        ],
    },
    {
        "ref": "[14]",
        "filename": "14_baena_garcia_2006_eddm.pdf",
        "title": "Early Drift Detection Method (EDDM)",
        "authors": "Baena-Garcia et al. (2006)",
        "urls": [
            "http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.9085&rep=rep1&type=pdf",
            "https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=10.1.1.58.9085",
        ],
    },
    {
        "ref": "[15]",
        "filename": "15_barros_2017_rddm.pdf",
        "title": "RDDM: Reactive Drift Detection Method",
        "authors": "Barros et al. (2017)",
        "urls": [
            "https://www.researchgate.net/profile/Roberto-Barros-3/publication/319166247_RDDM_Reactive_drift_detection_method/links/59a117e2aca272895c175d44/RDDM-Reactive-drift-detection-method.pdf",
        ],
    },
    {
        "ref": "[16]",
        "filename": "16_sculley_2015_hidden_technical_debt_ml.pdf",
        "title": "Hidden Technical Debt in Machine Learning Systems",
        "authors": "Sculley et al. (2015)",
        "urls": [
            "https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf",
            "https://papers.nips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf",
        ],
    },
    {
        "ref": "[18]",
        "filename": "18_sebastiao_2017_page_hinkley_emd.pdf",
        "title": "Supporting the Page-Hinkley Test with EMD for Change Detection",
        "authors": "Sebastiao & Fernandes (2017)",
        "urls": [
            "https://www.researchgate.net/profile/Raquel-Sebastiao/publication/318181504_Supporting_the_Page-Hinkley_Test_with_Empirical_Mode_Decomposition_for_Change_Detection/links/595e86d1aca272d2e9d0d7d0/Supporting-the-Page-Hinkley-Test-with-Empirical-Mode-Decomposition-for-Change-Detection.pdf",
        ],
    },
    {
        "ref": "[19]",
        "filename": "19_gretton_2012_kernel_two_sample_test.pdf",
        "title": "A Kernel Two-Sample Test",
        "authors": "Gretton et al. (2012)",
        "urls": [
            "https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf",
        ],
    },
    {
        "ref": "[20]",
        "filename": "20_losing_2018_incremental_online_learning.pdf",
        "title": "Incremental On-line Learning: A Review and Comparison",
        "authors": "Losing et al. (2018)",
        "urls": [
            "https://arxiv.org/pdf/1802.02871.pdf",
        ],
    },
    {
        "ref": "[22]",
        "filename": "22_webb_2016_characterizing_concept_drift.pdf",
        "title": "Characterizing Concept Drift",
        "authors": "Webb et al. (2016)",
        "urls": [
            "https://arxiv.org/pdf/1511.03816.pdf",
        ],
    },
    {
        "ref": "[23]",
        "filename": "23_zliobaite_2010_concept_drift_overview.pdf",
        "title": "Learning Under Concept Drift: An Overview",
        "authors": "Zliobaite (2010)",
        "urls": [
            "https://arxiv.org/pdf/1010.4784.pdf",
        ],
    },
    {
        "ref": "[24]",
        "filename": "24_bifet_2010_moa.pdf",
        "title": "MOA: Massive Online Analysis",
        "authors": "Bifet et al. (2010)",
        "urls": [
            "https://jmlr.org/papers/volume11/bifet10a/bifet10a.pdf",
        ],
    },
    {
        "ref": "[25]",
        "filename": "25_gozuacik_2021_one_class_drift_detection.pdf",
        "title": "Concept Learning Using One-Class Classifiers for Drift Detection",
        "authors": "Gozuacik & Can (2021)",
        "urls": [
            "https://arxiv.org/pdf/1907.09525.pdf",
        ],
    },
]

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0 Safari/537.36"


def curl_download(url, filepath):
    """Download url to filepath using curl. Returns (success, size_kb, error)."""
    cmd = [
        "curl", "-L", "-s", "--max-time", "30",
        "-A", UA,
        "-o", filepath,
        "-w", "%{http_code}",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    http_code = result.stdout.strip()
    if result.returncode != 0:
        return False, 0, f"curl error {result.returncode}: {result.stderr[:120]}"
    if http_code not in ("200", "206"):
        return False, 0, f"HTTP {http_code}"
    return True, 0, ""


def is_valid_pdf(filepath):
    """Check file starts with %PDF- and is at least 20 KB."""
    try:
        size = os.path.getsize(filepath)
        if size < 20_000:
            return False, size // 1024
        with open(filepath, "rb") as f:
            ok = f.read(5) == b"%PDF-"
        return ok, size // 1024
    except Exception:
        return False, 0


results = []

for paper in PAPERS:
    ref        = paper["ref"]
    filename   = paper["filename"]
    title      = paper["title"]
    authors    = paper["authors"]
    urls       = paper.get("urls", [])
    note       = paper.get("note", "")

    print(f"\n{'='*62}")
    print(f"{ref}  {authors}")
    print(f"     {title[:70]}")

    if not urls:
        print(f"     [SKIP] {note}")
        results.append((ref, authors, title, filename, "SKIP", note))
        continue

    filepath = os.path.join(PAPERS_DIR, filename)

    # Already a good file?
    if os.path.exists(filepath):
        ok, kb = is_valid_pdf(filepath)
        if ok:
            print(f"     [OK] Already downloaded ({kb} KB)")
            results.append((ref, authors, title, filename, f"OK ({kb} KB)", "Already present"))
            continue
        else:
            os.remove(filepath)

    success = False
    for i, url in enumerate(urls, 1):
        short_url = url[:80] + ("..." if len(url) > 80 else "")
        print(f"     [{i}/{len(urls)}] {short_url}")
        ok_dl, _, err = curl_download(url, filepath)
        if ok_dl:
            valid, kb = is_valid_pdf(filepath)
            if valid:
                print(f"     [OK] {kb} KB")
                results.append((ref, authors, title, filename, f"OK ({kb} KB)", url[:80]))
                success = True
                break
            else:
                print(f"     [INVALID] Not a PDF (HTML/redirect) — {kb} KB")
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            print(f"     [FAIL] {err}")
            if os.path.exists(filepath):
                os.remove(filepath)
        time.sleep(0.8)

    if not success:
        reason = note or "All URLs failed / paywalled"
        print(f"     [FAIL] {reason}")
        results.append((ref, authors, title, "—", "FAIL", reason))

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
ok_count   = sum(1 for r in results if r[4].startswith("OK"))
skip_count = sum(1 for r in results if r[4] == "SKIP")
fail_count = sum(1 for r in results if r[4] == "FAIL")

print(f"\n{'='*62}")
print(f"COMPLETE: {ok_count} downloaded  {skip_count} skipped (paywalled)  {fail_count} failed")

# --------------------------------------------------------------------------
# Write log
# --------------------------------------------------------------------------
log_path = os.path.join(PAPERS_DIR, "DOWNLOAD_LOG.md")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("# Research Papers — Download Log\n\n")
    f.write(f"**Papers attempted:** {len(PAPERS)}  \n")
    f.write(f"**Downloaded:** {ok_count}  |  ")
    f.write(f"**Skipped (paywalled):** {skip_count}  |  ")
    f.write(f"**Failed:** {fail_count}\n\n")
    f.write("> Papers marked SKIP are protected by publisher paywalls (JSTOR, ")
    f.write("Biometrika, JASA). Access them via your institutional library.\n\n")
    f.write("---\n\n")
    f.write("| Ref | Authors | Title | File | Status |\n")
    f.write("|-----|---------|-------|------|--------|\n")
    for ref, authors, title, fname, status, _ in results:
        t = title[:55] + "..." if len(title) > 55 else title
        icon = "✅" if status.startswith("OK") else ("⚠️" if status == "SKIP" else "❌")
        f.write(f"| {ref} | {authors} | {t} | `{fname}` | {icon} {status} |\n")

print(f"Log: {log_path}")
