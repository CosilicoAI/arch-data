"""Export targets database to JSON for frontend consumption."""

import json
from pathlib import Path
from collections import defaultdict

from sqlmodel import Session, select

from db.schema import (
    Stratum,
    Target,
    get_engine,
    DEFAULT_DB_PATH,
)


def export_targets_summary(output_path: Path | None = None):
    """Export summary statistics for the calibration dashboard."""

    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / "cosilico.ai" / "public" / "data" / "targets_summary.json"

    engine = get_engine(DEFAULT_DB_PATH)

    with Session(engine) as session:
        targets = session.exec(select(Target)).all()
        strata = session.exec(select(Stratum)).all()

        # Build summary
        by_source = defaultdict(lambda: {"count": 0, "years": set(), "variables": set()})
        by_year = defaultdict(lambda: {"count": 0, "sources": set()})
        by_jurisdiction = defaultdict(lambda: {"count": 0, "sources": set()})

        # Get stratum jurisdiction lookup
        stratum_jurisdiction = {s.id: s.jurisdiction.value for s in strata}

        for t in targets:
            source = t.source.value
            year = t.period
            jurisdiction = stratum_jurisdiction.get(t.stratum_id, "unknown")

            by_source[source]["count"] += 1
            by_source[source]["years"].add(year)
            by_source[source]["variables"].add(t.variable)

            by_year[year]["count"] += 1
            by_year[year]["sources"].add(source)

            by_jurisdiction[jurisdiction]["count"] += 1
            by_jurisdiction[jurisdiction]["sources"].add(source)

        # Convert sets to lists for JSON
        sources_summary = []
        for source, data in by_source.items():
            years = sorted(data["years"])
            sources_summary.append({
                "source": source,
                "display_name": source.upper().replace("-", " ").replace("_", " "),
                "count": data["count"],
                "variables": len(data["variables"]),
                "year_min": min(years) if years else None,
                "year_max": max(years) if years else None,
                "years": years,
                "is_projection": min(years) >= 2024 if years else False,
            })

        # Sort: historical sources first, then projections
        sources_summary.sort(key=lambda x: (x["is_projection"], x["source"]))

        years_summary = [
            {
                "year": year,
                "count": data["count"],
                "sources": sorted(data["sources"]),
            }
            for year, data in sorted(by_year.items())
        ]

        jurisdictions_summary = [
            {
                "jurisdiction": jurisdiction,
                "count": data["count"],
                "sources": sorted(data["sources"]),
            }
            for jurisdiction, data in sorted(by_jurisdiction.items())
        ]

        # Sample targets for display
        sample_targets = []
        for t in targets[:50]:
            stratum = session.get(Stratum, t.stratum_id)
            sample_targets.append({
                "stratum": stratum.name if stratum else "Unknown",
                "variable": t.variable,
                "year": t.period,
                "value": t.value,
                "type": t.target_type.value,
                "source": t.source.value,
            })

        summary = {
            "generated_at": str(Path(DEFAULT_DB_PATH).stat().st_mtime),
            "total_targets": len(targets),
            "total_strata": len(strata),
            "sources": sources_summary,
            "years": years_summary,
            "jurisdictions": jurisdictions_summary,
            "sample_targets": sample_targets,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Exported summary to {output_path}")
        print(f"  {len(targets)} targets across {len(strata)} strata")
        print(f"  {len(sources_summary)} sources: {', '.join(s['source'] for s in sources_summary)}")

        return summary


if __name__ == "__main__":
    export_targets_summary()
