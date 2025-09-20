"""
MiFID II Regulatory Compliance Automation System.

Implements comprehensive compliance monitoring, reporting, and audit trail
generation for MiFID II regulations including best execution, transaction
reporting, and client categorization.
"""
from __future__ import annotations

import uuid
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from decimal import Decimal

logger = logging.getLogger(__name__)


class ClientCategory(Enum):
    """MiFID II client categorization."""
    RETAIL = "retail"
    PROFESSIONAL = "professional"
    ELIGIBLE_COUNTERPARTY = "eligible_counterparty"


class OrderType(Enum):
    """Order types for compliance tracking."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    ALGORITHMIC = "algorithmic"


class ExecutionVenue(Enum):
    """Execution venues for best execution."""
    PRIMARY_EXCHANGE = "primary_exchange"
    ALTERNATIVE_EXCHANGE = "alternative_exchange"
    DARK_POOL = "dark_pool"
    SYSTEMATIC_INTERNALISER = "systematic_internaliser"
    OTC = "otc"
    MTF = "multilateral_trading_facility"
    OTF = "organised_trading_facility"


class TransactionReportField(Enum):
    """Required fields for MiFID II transaction reporting."""
    EXECUTING_ENTITY = "executing_entity"
    BUYER_ID = "buyer_id"
    SELLER_ID = "seller_id"
    TRADING_DATE = "trading_date"
    TRADING_TIME = "trading_time"
    INSTRUMENT_ID = "instrument_id"
    PRICE = "price"
    QUANTITY = "quantity"
    VENUE = "venue"
    REFERENCE_NUMBER = "reference_number"


@dataclass
class ComplianceCheck:
    """Individual compliance check result."""
    check_id: str
    check_type: str
    status: str  # passed, failed, warning
    description: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    remediation: Optional[str] = None


@dataclass
class BestExecutionAnalysis:
    """Best execution analysis for order."""
    order_id: str
    execution_quality_score: float  # 0-100
    price_improvement: float
    execution_speed_ms: float
    venue_analysis: Dict[ExecutionVenue, float]
    factors_considered: Dict[str, float]
    compliance_status: str
    timestamp: datetime


@dataclass
class TransactionReport:
    """MiFID II transaction report."""
    report_id: str
    transaction_reference: str
    executing_entity: str
    buyer_id: str
    seller_id: str
    instrument_id: str
    isin: str
    quantity: Decimal
    price: Decimal
    venue: ExecutionVenue
    trading_date: datetime
    trading_time: datetime
    client_category: ClientCategory
    algo_indicator: bool
    short_selling_indicator: bool
    waiver_indicator: Optional[str]
    commodity_derivative_indicator: bool
    securities_financing_transaction: bool
    status: str  # pending, submitted, accepted, rejected
    arm_flags: Dict[str, bool] = field(default_factory=dict)  # Approved Reporting Mechanism


@dataclass
class AuditEntry:
    """Immutable audit trail entry."""
    audit_id: str
    event_type: str
    entity_id: str
    user_id: str
    timestamp: datetime
    action: str
    details: Dict[str, Any]
    hash_previous: str
    hash_current: str
    ip_address: Optional[str] = None
    compliance_checks: List[ComplianceCheck] = field(default_factory=list)


class BestExecutionMonitor:
    """Monitor and validate best execution compliance."""

    def __init__(self):
        self.execution_history: List[BestExecutionAnalysis] = []
        self.venue_statistics: Dict[ExecutionVenue, Dict[str, float]] = {}
        self._initialize_venue_stats()

    def _initialize_venue_stats(self):
        """Initialize venue statistics tracking."""
        for venue in ExecutionVenue:
            self.venue_statistics[venue] = {
                "total_volume": 0.0,
                "average_price_improvement": 0.0,
                "average_execution_speed": 0.0,
                "success_rate": 0.0,
                "total_trades": 0
            }

    async def analyze_execution(
        self,
        order_id: str,
        executed_price: float,
        executed_quantity: float,
        venue: ExecutionVenue,
        execution_time_ms: float,
        market_prices: Dict[ExecutionVenue, float],
        client_category: ClientCategory
    ) -> BestExecutionAnalysis:
        """Analyze order execution for best execution compliance."""

        # Calculate best available price
        best_price = min(market_prices.values())
        price_improvement = best_price - executed_price

        # Calculate execution quality score
        quality_score = self._calculate_quality_score(
            executed_price,
            best_price,
            execution_time_ms,
            client_category
        )

        # Analyze venue selection
        venue_analysis = self._analyze_venue_selection(
            market_prices,
            executed_price,
            execution_time_ms
        )

        # Determine compliance status
        compliance_status = "compliant" if quality_score >= 85 else "review_required"

        # Consider MiFID II factors
        factors_considered = {
            "price": self._score_price_factor(executed_price, best_price),
            "costs": self._estimate_total_costs(venue, executed_quantity),
            "speed": self._score_speed_factor(execution_time_ms),
            "likelihood_of_execution": self._estimate_execution_likelihood(venue),
            "size": executed_quantity,
            "nature": 1.0 if client_category == ClientCategory.RETAIL else 0.5
        }

        analysis = BestExecutionAnalysis(
            order_id=order_id,
            execution_quality_score=quality_score,
            price_improvement=price_improvement,
            execution_speed_ms=execution_time_ms,
            venue_analysis=venue_analysis,
            factors_considered=factors_considered,
            compliance_status=compliance_status,
            timestamp=datetime.now(timezone.utc)
        )

        # Update venue statistics
        self._update_venue_statistics(venue, analysis)
        self.execution_history.append(analysis)

        logger.info(f"Best execution analysis for {order_id}: Score={quality_score:.2f}, Status={compliance_status}")

        return analysis

    def _calculate_quality_score(
        self,
        executed_price: float,
        best_price: float,
        execution_time_ms: float,
        client_category: ClientCategory
    ) -> float:
        """Calculate overall execution quality score."""
        # Price component (40% weight)
        price_score = max(0, 100 * (1 - abs(executed_price - best_price) / best_price))

        # Speed component (30% weight)
        speed_score = max(0, 100 * (1 - execution_time_ms / 1000))  # Target <1 second

        # Client category adjustment (30% weight)
        if client_category == ClientCategory.RETAIL:
            category_score = 100  # Highest protection
        elif client_category == ClientCategory.PROFESSIONAL:
            category_score = 75
        else:
            category_score = 50

        quality_score = (
            0.4 * price_score +
            0.3 * speed_score +
            0.3 * category_score
        )

        return min(100, max(0, quality_score))

    def _analyze_venue_selection(
        self,
        market_prices: Dict[ExecutionVenue, float],
        executed_price: float,
        execution_time_ms: float
    ) -> Dict[ExecutionVenue, float]:
        """Analyze venue selection decision."""
        venue_scores = {}

        for venue, price in market_prices.items():
            # Score based on price and historical performance
            price_score = 100 * (1 - abs(price - executed_price) / executed_price)

            historical_score = 50  # Default
            if venue in self.venue_statistics:
                stats = self.venue_statistics[venue]
                if stats["total_trades"] > 0:
                    historical_score = stats["success_rate"]

            venue_scores[venue] = 0.7 * price_score + 0.3 * historical_score

        return venue_scores

    def _score_price_factor(self, executed_price: float, best_price: float) -> float:
        """Score the price factor for best execution."""
        if best_price == 0:
            return 0.0
        return max(0, 1 - abs(executed_price - best_price) / best_price)

    def _estimate_total_costs(self, venue: ExecutionVenue, quantity: float) -> float:
        """Estimate total costs including fees and market impact."""
        base_costs = {
            ExecutionVenue.PRIMARY_EXCHANGE: 0.001,
            ExecutionVenue.ALTERNATIVE_EXCHANGE: 0.0008,
            ExecutionVenue.DARK_POOL: 0.0005,
            ExecutionVenue.SYSTEMATIC_INTERNALISER: 0.0003,
            ExecutionVenue.OTC: 0.002,
            ExecutionVenue.MTF: 0.0007,
            ExecutionVenue.OTF: 0.0009
        }

        # Base cost + market impact estimate
        base_cost = base_costs.get(venue, 0.001)
        market_impact = 0.0001 * (quantity / 10000)  # Simplified market impact

        return base_cost + market_impact

    def _score_speed_factor(self, execution_time_ms: float) -> float:
        """Score execution speed factor."""
        if execution_time_ms <= 10:
            return 1.0
        elif execution_time_ms <= 100:
            return 0.9
        elif execution_time_ms <= 500:
            return 0.7
        elif execution_time_ms <= 1000:
            return 0.5
        else:
            return 0.3

    def _estimate_execution_likelihood(self, venue: ExecutionVenue) -> float:
        """Estimate likelihood of execution at venue."""
        likelihood_scores = {
            ExecutionVenue.PRIMARY_EXCHANGE: 0.95,
            ExecutionVenue.ALTERNATIVE_EXCHANGE: 0.90,
            ExecutionVenue.DARK_POOL: 0.70,
            ExecutionVenue.SYSTEMATIC_INTERNALISER: 0.85,
            ExecutionVenue.OTC: 0.80,
            ExecutionVenue.MTF: 0.88,
            ExecutionVenue.OTF: 0.82
        }
        return likelihood_scores.get(venue, 0.75)

    def _update_venue_statistics(self, venue: ExecutionVenue, analysis: BestExecutionAnalysis):
        """Update venue performance statistics."""
        stats = self.venue_statistics[venue]

        # Update running averages
        n = stats["total_trades"]
        stats["average_price_improvement"] = (
            (stats["average_price_improvement"] * n + analysis.price_improvement) / (n + 1)
        )
        stats["average_execution_speed"] = (
            (stats["average_execution_speed"] * n + analysis.execution_speed_ms) / (n + 1)
        )

        # Update success rate
        if analysis.compliance_status == "compliant":
            stats["success_rate"] = (stats["success_rate"] * n + 100) / (n + 1)
        else:
            stats["success_rate"] = (stats["success_rate"] * n) / (n + 1)

        stats["total_trades"] += 1

    def generate_quarterly_report(self) -> Dict[str, Any]:
        """Generate quarterly best execution report."""
        if not self.execution_history:
            return {"status": "no_data"}

        # Filter last quarter
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
        quarterly_executions = [
            e for e in self.execution_history
            if e.timestamp >= cutoff_date
        ]

        if not quarterly_executions:
            return {"status": "no_recent_data"}

        # Calculate statistics
        total_executions = len(quarterly_executions)
        compliant_executions = sum(1 for e in quarterly_executions if e.compliance_status == "compliant")

        avg_quality_score = sum(e.execution_quality_score for e in quarterly_executions) / total_executions
        avg_price_improvement = sum(e.price_improvement for e in quarterly_executions) / total_executions
        avg_execution_speed = sum(e.execution_speed_ms for e in quarterly_executions) / total_executions

        # Venue breakdown
        venue_usage = {}
        for execution in quarterly_executions:
            for venue, score in execution.venue_analysis.items():
                if venue not in venue_usage:
                    venue_usage[venue] = {"count": 0, "total_score": 0}
                venue_usage[venue]["count"] += 1
                venue_usage[venue]["total_score"] += score

        venue_performance = {
            venue.value: {
                "usage_percentage": (data["count"] / total_executions) * 100,
                "average_score": data["total_score"] / data["count"]
            }
            for venue, data in venue_usage.items()
        }

        return {
            "period": {
                "start": cutoff_date.isoformat(),
                "end": datetime.now(timezone.utc).isoformat()
            },
            "total_executions": total_executions,
            "compliance_rate": (compliant_executions / total_executions) * 100,
            "average_quality_score": avg_quality_score,
            "average_price_improvement": avg_price_improvement,
            "average_execution_speed_ms": avg_execution_speed,
            "venue_performance": venue_performance,
            "venue_statistics": {
                venue.value: stats
                for venue, stats in self.venue_statistics.items()
                if stats["total_trades"] > 0
            }
        }


class TransactionReporter:
    """Handle MiFID II transaction reporting."""

    def __init__(self, firm_id: str, reporting_endpoint: str = "https://arm.regulator.eu/api/v1"):
        self.firm_id = firm_id
        self.reporting_endpoint = reporting_endpoint
        self.pending_reports: List[TransactionReport] = []
        self.submitted_reports: Dict[str, TransactionReport] = {}

    async def create_transaction_report(
        self,
        trade_data: Dict[str, Any],
        client_category: ClientCategory
    ) -> TransactionReport:
        """Create a MiFID II compliant transaction report."""

        # Generate unique reference
        transaction_ref = f"TRN-{self.firm_id}-{uuid.uuid4().hex[:12].upper()}"

        # Create report
        report = TransactionReport(
            report_id=str(uuid.uuid4()),
            transaction_reference=transaction_ref,
            executing_entity=self.firm_id,
            buyer_id=trade_data.get("buyer_id", ""),
            seller_id=trade_data.get("seller_id", ""),
            instrument_id=trade_data["instrument_id"],
            isin=trade_data.get("isin", ""),
            quantity=Decimal(str(trade_data["quantity"])),
            price=Decimal(str(trade_data["price"])),
            venue=ExecutionVenue(trade_data.get("venue", "primary_exchange")),
            trading_date=datetime.now(timezone.utc).date(),
            trading_time=datetime.now(timezone.utc),
            client_category=client_category,
            algo_indicator=trade_data.get("algo_indicator", False),
            short_selling_indicator=trade_data.get("short_selling", False),
            waiver_indicator=trade_data.get("waiver", None),
            commodity_derivative_indicator=trade_data.get("commodity_derivative", False),
            securities_financing_transaction=trade_data.get("sft", False),
            status="pending"
        )

        # Validate report
        validation_errors = self._validate_report(report)
        if validation_errors:
            logger.error(f"Transaction report validation failed: {validation_errors}")
            report.status = "validation_failed"
        else:
            self.pending_reports.append(report)

        return report

    def _validate_report(self, report: TransactionReport) -> List[str]:
        """Validate transaction report for MiFID II compliance."""
        errors = []

        # Check required fields
        if not report.executing_entity:
            errors.append("Missing executing entity")

        if not report.instrument_id and not report.isin:
            errors.append("Missing instrument identifier (ID or ISIN required)")

        if report.quantity <= 0:
            errors.append("Invalid quantity")

        if report.price <= 0:
            errors.append("Invalid price")

        # Check client categorization
        if report.client_category not in ClientCategory:
            errors.append("Invalid client category")

        # Check timestamps
        if report.trading_time > datetime.now(timezone.utc):
            errors.append("Trading time cannot be in the future")

        return errors

    async def submit_reports(self) -> Dict[str, Any]:
        """Submit pending transaction reports to ARM."""
        if not self.pending_reports:
            return {"status": "no_pending_reports"}

        submitted_count = 0
        failed_count = 0

        for report in self.pending_reports[:]:
            try:
                # Simulate ARM submission
                await self._submit_to_arm(report)

                report.status = "submitted"
                self.submitted_reports[report.report_id] = report
                self.pending_reports.remove(report)
                submitted_count += 1

            except Exception as e:
                logger.error(f"Failed to submit report {report.report_id}: {e}")
                report.status = "submission_failed"
                failed_count += 1

        return {
            "submitted": submitted_count,
            "failed": failed_count,
            "remaining_pending": len(self.pending_reports),
            "total_submitted": len(self.submitted_reports)
        }

    async def _submit_to_arm(self, report: TransactionReport):
        """Submit report to Approved Reporting Mechanism."""
        # Simulate ARM submission with delay
        await asyncio.sleep(0.1)

        # In production, would make actual API call to ARM
        logger.info(f"Report {report.transaction_reference} submitted to ARM")


class AuditTrailManager:
    """Immutable audit trail for compliance tracking."""

    def __init__(self):
        self.audit_trail: List[AuditEntry] = []
        self.last_hash = "0" * 64  # Initial hash
        self._lock = asyncio.Lock()

    async def add_entry(
        self,
        event_type: str,
        entity_id: str,
        user_id: str,
        action: str,
        details: Dict[str, Any],
        compliance_checks: Optional[List[ComplianceCheck]] = None
    ) -> AuditEntry:
        """Add immutable entry to audit trail."""

        async with self._lock:
            # Create hash of current entry
            entry_data = f"{event_type}{entity_id}{user_id}{action}{json.dumps(details, sort_keys=True)}"
            current_hash = hashlib.sha256(
                f"{self.last_hash}{entry_data}".encode()
            ).hexdigest()

            # Create audit entry
            entry = AuditEntry(
                audit_id=str(uuid.uuid4()),
                event_type=event_type,
                entity_id=entity_id,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc),
                action=action,
                details=details,
                hash_previous=self.last_hash,
                hash_current=current_hash,
                compliance_checks=compliance_checks or []
            )

            self.audit_trail.append(entry)
            self.last_hash = current_hash

            logger.info(f"Audit entry added: {event_type} - {action} by {user_id}")

            return entry

    def verify_integrity(self) -> bool:
        """Verify audit trail integrity."""
        if not self.audit_trail:
            return True

        previous_hash = "0" * 64

        for entry in self.audit_trail:
            # Verify previous hash matches
            if entry.hash_previous != previous_hash:
                logger.error(f"Audit trail integrity violation at {entry.audit_id}")
                return False

            # Recalculate hash
            entry_data = f"{entry.event_type}{entry.entity_id}{entry.user_id}{entry.action}{json.dumps(entry.details, sort_keys=True)}"
            calculated_hash = hashlib.sha256(
                f"{previous_hash}{entry_data}".encode()
            ).hexdigest()

            if calculated_hash != entry.hash_current:
                logger.error(f"Hash mismatch at audit entry {entry.audit_id}")
                return False

            previous_hash = entry.hash_current

        return True

    def get_audit_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate audit report for compliance review."""

        # Filter entries
        filtered_entries = self.audit_trail

        if start_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_date]

        if end_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_date]

        if event_type:
            filtered_entries = [e for e in filtered_entries if e.event_type == event_type]

        # Generate statistics
        event_types = {}
        compliance_stats = {"passed": 0, "failed": 0, "warning": 0}

        for entry in filtered_entries:
            # Count event types
            event_types[entry.event_type] = event_types.get(entry.event_type, 0) + 1

            # Count compliance check results
            for check in entry.compliance_checks:
                if check.status in compliance_stats:
                    compliance_stats[check.status] += 1

        return {
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None
            },
            "total_entries": len(filtered_entries),
            "event_types": event_types,
            "compliance_statistics": compliance_stats,
            "integrity_verified": self.verify_integrity(),
            "entries": [
                {
                    "audit_id": e.audit_id,
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type,
                    "action": e.action,
                    "user_id": e.user_id,
                    "compliance_checks": len(e.compliance_checks)
                }
                for e in filtered_entries[-100:]  # Last 100 entries
            ]
        }


class MiFIDComplianceManager:
    """Central MiFID II compliance management system."""

    def __init__(self, firm_id: str):
        self.firm_id = firm_id
        self.best_execution = BestExecutionMonitor()
        self.transaction_reporter = TransactionReporter(firm_id)
        self.audit_trail = AuditTrailManager()

    async def process_trade(
        self,
        trade_data: Dict[str, Any],
        client_category: ClientCategory,
        user_id: str
    ) -> Dict[str, Any]:
        """Process trade through compliance pipeline."""

        trade_id = trade_data.get("trade_id", str(uuid.uuid4()))

        # Perform compliance checks
        compliance_checks = []

        # Check 1: Client categorization
        cat_check = ComplianceCheck(
            check_id=str(uuid.uuid4()),
            check_type="client_categorization",
            status="passed" if client_category in ClientCategory else "failed",
            description="Verify client category is properly assigned",
            timestamp=datetime.now(timezone.utc)
        )
        compliance_checks.append(cat_check)

        # Check 2: Best execution
        if "market_prices" in trade_data:
            best_exec_analysis = await self.best_execution.analyze_execution(
                order_id=trade_id,
                executed_price=trade_data["price"],
                executed_quantity=trade_data["quantity"],
                venue=ExecutionVenue(trade_data.get("venue", "primary_exchange")),
                execution_time_ms=trade_data.get("execution_time_ms", 100),
                market_prices=trade_data["market_prices"],
                client_category=client_category
            )

            exec_check = ComplianceCheck(
                check_id=str(uuid.uuid4()),
                check_type="best_execution",
                status="passed" if best_exec_analysis.compliance_status == "compliant" else "warning",
                description=f"Best execution score: {best_exec_analysis.execution_quality_score:.2f}",
                timestamp=datetime.now(timezone.utc),
                details={"analysis": best_exec_analysis.__dict__}
            )
            compliance_checks.append(exec_check)

        # Check 3: Transaction reporting
        transaction_report = await self.transaction_reporter.create_transaction_report(
            trade_data, client_category
        )

        report_check = ComplianceCheck(
            check_id=str(uuid.uuid4()),
            check_type="transaction_reporting",
            status="passed" if transaction_report.status != "validation_failed" else "failed",
            description="Transaction report created for regulatory submission",
            timestamp=datetime.now(timezone.utc),
            details={"report_id": transaction_report.report_id}
        )
        compliance_checks.append(report_check)

        # Add to audit trail
        audit_entry = await self.audit_trail.add_entry(
            event_type="trade_execution",
            entity_id=trade_id,
            user_id=user_id,
            action="process_trade",
            details=trade_data,
            compliance_checks=compliance_checks
        )

        # Determine overall compliance status
        failed_checks = [c for c in compliance_checks if c.status == "failed"]
        warning_checks = [c for c in compliance_checks if c.status == "warning"]

        if failed_checks:
            compliance_status = "non_compliant"
        elif warning_checks:
            compliance_status = "review_required"
        else:
            compliance_status = "compliant"

        return {
            "trade_id": trade_id,
            "compliance_status": compliance_status,
            "checks_performed": len(compliance_checks),
            "failed_checks": len(failed_checks),
            "warning_checks": len(warning_checks),
            "transaction_report_id": transaction_report.report_id,
            "audit_id": audit_entry.audit_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def generate_regulatory_reports(self) -> Dict[str, Any]:
        """Generate all required MiFID II regulatory reports."""

        # Best execution quarterly report
        best_exec_report = self.best_execution.generate_quarterly_report()

        # Transaction reporting status
        submission_status = await self.transaction_reporter.submit_reports()

        # Audit trail report
        audit_report = self.audit_trail.get_audit_report(
            start_date=datetime.now(timezone.utc) - timedelta(days=90)
        )

        return {
            "report_date": datetime.now(timezone.utc).isoformat(),
            "firm_id": self.firm_id,
            "best_execution": best_exec_report,
            "transaction_reporting": submission_status,
            "audit_trail": audit_report,
            "compliance_summary": {
                "total_trades": best_exec_report.get("total_executions", 0),
                "compliance_rate": best_exec_report.get("compliance_rate", 0),
                "reports_submitted": submission_status.get("submitted", 0),
                "audit_entries": audit_report.get("total_entries", 0),
                "integrity_verified": audit_report.get("integrity_verified", False)
            }
        }


__all__ = [
    "MiFIDComplianceManager",
    "BestExecutionMonitor",
    "TransactionReporter",
    "AuditTrailManager",
    "ClientCategory",
    "OrderType",
    "ExecutionVenue",
    "ComplianceCheck",
    "BestExecutionAnalysis",
    "TransactionReport",
    "AuditEntry"
]