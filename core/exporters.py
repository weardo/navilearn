"""CSV exporters for student dashboard data.

Thin, dependency-free serializers that turn repository output into CSV text.
They use the stdlib :mod:`csv` module (via :class:`io.StringIO`) so they carry
no Arrow/pandas dependency and can back both a Streamlit ``download_button`` and
the REST API without change.

Each function returns a ready-to-download CSV *string*; callers own the
filename and the ``text/csv`` mime type.
"""

from __future__ import annotations

import csv
import io

from core.repo import Repository


def progress_csv(repo: Repository, student_id: str) -> str:
    """Return the student's per-course progress as CSV text.

    Columns: ``course``, ``completed``, ``total``, ``pct``. One row per course
    as returned by :meth:`Repository.progress_by_course`. The header is always
    written, so an empty dataset yields a header-only CSV rather than an empty
    string.
    """

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["course", "completed", "total", "pct"])
    for row in repo.progress_by_course(student_id):
        writer.writerow(
            [
                row.get("course", ""),
                row.get("completed", 0),
                row.get("total", 0),
                row.get("pct", 0.0),
            ]
        )
    return buffer.getvalue()


def activity_csv(repo: Repository, student_id: str, days: int = 30) -> str:
    """Return the student's daily activity time series as CSV text.

    Columns: ``date``, ``minutes``. One row per day over the last ``days`` days
    as returned by :meth:`Repository.activity_timeseries` (zero-filled for rest
    days). The header is always written.
    """

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["date", "minutes"])
    for point in repo.activity_timeseries(student_id, days=days):
        writer.writerow([point.get("date", ""), point.get("minutes", 0.0)])
    return buffer.getvalue()
