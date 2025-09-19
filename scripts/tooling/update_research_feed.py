#!/usr/bin/env python3
"""Utility to append or update research insights in data/processed/research_feed.json."""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

FEED_PATH = Path("data/processed/research_feed.json")
SOURCES_PATH = Path("agents/configs/innovation_sources.yaml")


def load_feed() -> Dict[str, Any]:
    if not FEED_PATH.exists():
        return {"generated_at": datetime.now(timezone.utc).isoformat(), "insights": []}
    return json.loads(FEED_PATH.read_text(encoding="utf-8"))


def save_feed(feed: Dict[str, Any]) -> None:
    feed["generated_at"] = datetime.now(timezone.utc).isoformat()
    FEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEED_PATH.write_text(json.dumps(feed, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update research feed with new insight")
    parser.add_argument("source_id", help="Source identifier defined in innovation_sources.yaml")
    parser.add_argument("title", help="Title of the insight")
    parser.add_argument("summary", help="Short summary")
    parser.add_argument("category", choices=["strategy", "risk", "execution", "data", "ops"], help="Opportunity category")
    parser.add_argument("impact", choices=["low", "medium", "high"], help="Impact estimate")
    parser.add_argument("effort", choices=["low", "medium", "high"], help="Effort estimate")
    parser.add_argument("provenance_url", help="URL to original source")
    parser.add_argument("--status", default="pending-validation", choices=["pending-validation", "queued", "in-validation", "adopted", "rejected"], help="Workflow status")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feed = load_feed()
    insight = {
        "id": f"insight-{uuid.uuid4().hex[:8]}",
        "source_id": args.source_id,
        "title": args.title,
        "summary": args.summary,
        "provenance_url": args.provenance_url,
        "category": args.category,
        "impact_estimate": args.impact,
        "effort_estimate": args.effort,
        "status": args.status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    feed.setdefault("insights", []).append(insight)
    save_feed(feed)
    print(f"Added insight {insight['id']} for source {args.source_id}")


if __name__ == "__main__":
    main()
