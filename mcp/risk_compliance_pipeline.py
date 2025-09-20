"""
Comprehensive Risk and Compliance Pipeline with Audit Capabilities.

This module creates a complete risk management and compliance framework
that integrates with the multi-agent orchestration system and provides
real-time monitoring, audit trails, and regulatory compliance.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    PENDING = "pending"


class AuditEventType(Enum):
    """Types of audit events."""
    TRADE_DECISION = "trade_decision"
    RISK_BREACH = "risk_breach"
    COMPLIANCE_CHECK = "compliance_check"
    LIMIT_CHANGE = "limit_change"
    AGENT_ACTION = "agent_action"
    SYSTEM_ALERT = "system_alert"


@dataclass
class RiskMetrics:
    """Risk assessment metrics."""
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    beta: float
    volatility: float
    concentration_risk: float
    liquidity_risk: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "expected_shortfall": self.expected_shortfall,
            "max_drawdown": self.max_drawdown,
            "beta": self.beta,
            "volatility": self.volatility,
            "concentration_risk": self.concentration_risk,
            "liquidity_risk": self.liquidity_risk
        }


@dataclass
class RiskLimits:
    """Risk limit definitions."""
    max_var_95: float = 0.05  # 5% of portfolio
    max_var_99: float = 0.10  # 10% of portfolio
    max_drawdown: float = 0.15  # 15% max drawdown
    max_concentration: float = 0.10  # 10% per position
    max_beta: float = 1.5
    max_volatility: float = 0.30  # 30% annualized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_var_95": self.max_var_95,
            "max_var_99": self.max_var_99,
            "max_drawdown": self.max_drawdown,
            "max_concentration": self.max_concentration,
            "max_beta": self.max_beta,
            "max_volatility": self.max_volatility
        }


@dataclass
class ComplianceRule:
    """Compliance rule definition."""
    rule_id: str
    rule_name: str
    description: str
    regulation: str  # e.g., "MiFID II", "SEC", "CFTC"
    check_function: str  # Function name to execute
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: ComplianceStatus = ComplianceStatus.WARNING
    enabled: bool = True


@dataclass
class AuditEvent:
    """Audit trail event."""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    risk_impact: Optional[RiskLevel] = None
    compliance_impact: Optional[ComplianceStatus] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "description": self.description,
            "details": self.details,
            "risk_impact": self.risk_impact.value if self.risk_impact else None,
            "compliance_impact": self.compliance_impact.value if self.compliance_impact else None
        }


@dataclass
class RiskAssessmentResult:
    """Result of risk assessment."""
    assessment_id: str
    timestamp: datetime
    metrics: RiskMetrics
    limit_breaches: List[str]
    risk_level: RiskLevel
    recommendations: List[str]
    approved: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics.to_dict(),
            "limit_breaches": self.limit_breaches,
            "risk_level": self.risk_level.value,
            "recommendations": self.recommendations,
            "approved": self.approved
        }


@dataclass
class ComplianceCheckResult:
    """Result of compliance check."""
    check_id: str
    timestamp: datetime
    rules_checked: List[str]
    violations: List[str]
    warnings: List[str]
    status: ComplianceStatus
    remediation_required: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "timestamp": self.timestamp.isoformat(),
            "rules_checked": self.rules_checked,
            "violations": self.violations,
            "warnings": self.warnings,
            "status": self.status.value,
            "remediation_required": self.remediation_required
        }


class RiskCompliancePipeline:
    """Comprehensive risk and compliance pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pipeline_id = str(uuid.uuid4())

        # Risk management
        self.risk_limits = RiskLimits(**self.config.get("risk_limits", {}))
        self.risk_assessments: Dict[str, RiskAssessmentResult] = {}

        # Compliance
        self.compliance_rules = self._load_compliance_rules()
        self.compliance_checks: Dict[str, ComplianceCheckResult] = {}

        # Audit trail
        self.audit_events: List[AuditEvent] = []
        self.audit_file = Path(self.config.get("audit_file", "audit_trail.jsonl"))

        # Monitoring
        self.monitoring_active = False
        self.alert_thresholds = self.config.get("alert_thresholds", {})

        logger.info(f"Risk/Compliance pipeline initialized: {self.pipeline_id}")

    def _load_compliance_rules(self) -> List[ComplianceRule]:
        """Load compliance rules from configuration."""
        rules = [
            ComplianceRule(
                rule_id="position_limit",
                rule_name="Maximum Position Size",
                description="No single position shall exceed 10% of portfolio value",
                regulation="Internal Policy",
                check_function="check_position_limits",
                parameters={"max_position_pct": 0.10}
            ),
            ComplianceRule(
                rule_id="leverage_limit",
                rule_name="Maximum Leverage",
                description="Total leverage shall not exceed 2:1",
                regulation="SEC Regulation T",
                check_function="check_leverage_limits",
                parameters={"max_leverage": 2.0}
            ),
            ComplianceRule(
                rule_id="wash_sale",
                rule_name="Wash Sale Prevention",
                description="Prevent wash sale violations per IRS regulations",
                regulation="IRS Section 1091",
                check_function="check_wash_sales",
                parameters={"lookback_days": 30}
            ),
            ComplianceRule(
                rule_id="best_execution",
                rule_name="Best Execution Requirement",
                description="All trades must seek best execution",
                regulation="MiFID II Article 27",
                check_function="check_best_execution",
                parameters={"price_improvement_threshold": 0.001}
            ),
            ComplianceRule(
                rule_id="market_abuse",
                rule_name="Market Abuse Prevention",
                description="Detect potential market manipulation patterns",
                regulation="EU Market Abuse Regulation",
                check_function="check_market_abuse",
                parameters={"volume_threshold": 0.05, "price_impact_threshold": 0.02}
            )
        ]

        # Add custom rules from config
        custom_rules = self.config.get("compliance_rules", [])
        for rule_config in custom_rules:
            rules.append(ComplianceRule(**rule_config))

        return rules

    async def assess_risk(
        self,
        portfolio_data: Dict[str, Any],
        proposed_trades: List[Dict[str, Any]] = None
    ) -> RiskAssessmentResult:
        """Perform comprehensive risk assessment."""
        assessment_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        try:
            # Calculate risk metrics
            metrics = await self._calculate_risk_metrics(portfolio_data, proposed_trades)

            # Check limit breaches
            limit_breaches = self._check_risk_limits(metrics)

            # Determine overall risk level
            risk_level = self._determine_risk_level(metrics, limit_breaches)

            # Generate recommendations
            recommendations = self._generate_risk_recommendations(metrics, limit_breaches)

            # Determine approval
            approved = len(limit_breaches) == 0 and risk_level != RiskLevel.CRITICAL

            result = RiskAssessmentResult(
                assessment_id=assessment_id,
                timestamp=timestamp,
                metrics=metrics,
                limit_breaches=limit_breaches,
                risk_level=risk_level,
                recommendations=recommendations,
                approved=approved
            )

            # Store assessment
            self.risk_assessments[assessment_id] = result

            # Log audit event
            await self._log_audit_event(
                AuditEventType.RISK_BREACH if limit_breaches else AuditEventType.TRADE_DECISION,
                f"Risk assessment completed: {risk_level.value}",
                {"assessment_id": assessment_id, "approved": approved},
                risk_impact=risk_level
            )

            return result

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            # Return conservative assessment
            return RiskAssessmentResult(
                assessment_id=assessment_id,
                timestamp=timestamp,
                metrics=RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0),
                limit_breaches=[f"Assessment error: {str(e)}"],
                risk_level=RiskLevel.CRITICAL,
                recommendations=["Manual review required due to assessment error"],
                approved=False
            )

    async def _calculate_risk_metrics(
        self,
        portfolio_data: Dict[str, Any],
        proposed_trades: List[Dict[str, Any]] = None
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        # This would normally use real portfolio data and market data
        # For demo purposes, using simplified calculations

        positions = portfolio_data.get("positions", [])
        total_value = sum(pos.get("market_value", 0) for pos in positions)

        if total_value == 0:
            return RiskMetrics(0, 0, 0, 0, 1.0, 0, 0, 0)

        # Simplified VaR calculation (normally would use historical simulation or Monte Carlo)
        volatilities = [pos.get("volatility", 0.2) for pos in positions]
        weights = [pos.get("market_value", 0) / total_value for pos in positions]

        portfolio_vol = sum(w * v for w, v in zip(weights, volatilities))
        var_95 = 1.65 * portfolio_vol * (total_value ** 0.5) / total_value  # Simplified
        var_99 = 2.33 * portfolio_vol * (total_value ** 0.5) / total_value

        # Concentration risk
        max_position = max(weights) if weights else 0
        concentration_risk = max_position

        # Beta (simplified as portfolio-weighted average)
        betas = [pos.get("beta", 1.0) for pos in positions]
        portfolio_beta = sum(w * b for w, b in zip(weights, betas))

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=var_99 * 1.2,  # Simplified ES
            max_drawdown=var_95 * 2,  # Simplified
            beta=portfolio_beta,
            volatility=portfolio_vol,
            concentration_risk=concentration_risk,
            liquidity_risk=0.05  # Simplified
        )

    def _check_risk_limits(self, metrics: RiskMetrics) -> List[str]:
        """Check metrics against defined limits."""
        breaches = []

        if metrics.var_95 > self.risk_limits.max_var_95:
            breaches.append(f"VaR 95% exceeds limit: {metrics.var_95:.2%} > {self.risk_limits.max_var_95:.2%}")

        if metrics.var_99 > self.risk_limits.max_var_99:
            breaches.append(f"VaR 99% exceeds limit: {metrics.var_99:.2%} > {self.risk_limits.max_var_99:.2%}")

        if metrics.max_drawdown > self.risk_limits.max_drawdown:
            breaches.append(f"Max drawdown exceeds limit: {metrics.max_drawdown:.2%} > {self.risk_limits.max_drawdown:.2%}")

        if metrics.concentration_risk > self.risk_limits.max_concentration:
            breaches.append(f"Concentration risk exceeds limit: {metrics.concentration_risk:.2%} > {self.risk_limits.max_concentration:.2%}")

        if abs(metrics.beta) > self.risk_limits.max_beta:
            breaches.append(f"Beta exceeds limit: {abs(metrics.beta):.2f} > {self.risk_limits.max_beta:.2f}")

        if metrics.volatility > self.risk_limits.max_volatility:
            breaches.append(f"Volatility exceeds limit: {metrics.volatility:.2%} > {self.risk_limits.max_volatility:.2%}")

        return breaches

    def _determine_risk_level(self, metrics: RiskMetrics, breaches: List[str]) -> RiskLevel:
        """Determine overall risk level."""
        if len(breaches) >= 3:
            return RiskLevel.CRITICAL
        elif len(breaches) >= 2:
            return RiskLevel.HIGH
        elif len(breaches) >= 1:
            return RiskLevel.MEDIUM
        elif metrics.var_95 > self.risk_limits.max_var_95 * 0.8:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _generate_risk_recommendations(self, metrics: RiskMetrics, breaches: List[str]) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []

        if metrics.var_95 > self.risk_limits.max_var_95:
            recommendations.append("Reduce portfolio leverage or hedge with offsetting positions")

        if metrics.concentration_risk > self.risk_limits.max_concentration:
            recommendations.append("Diversify holdings to reduce concentration risk")

        if metrics.volatility > self.risk_limits.max_volatility:
            recommendations.append("Consider adding lower-volatility assets to the portfolio")

        if abs(metrics.beta) > self.risk_limits.max_beta:
            if metrics.beta > 0:
                recommendations.append("Reduce market exposure or add market-neutral strategies")
            else:
                recommendations.append("Consider market exposure adjustment")

        if not recommendations:
            recommendations.append("Risk profile within acceptable limits")

        return recommendations

    async def check_compliance(
        self,
        trade_data: Dict[str, Any],
        portfolio_context: Dict[str, Any] = None
    ) -> ComplianceCheckResult:
        """Perform comprehensive compliance check."""
        check_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        violations = []
        warnings = []
        rules_checked = []

        for rule in self.compliance_rules:
            if not rule.enabled:
                continue

            rules_checked.append(rule.rule_id)

            try:
                # Execute compliance check
                check_result = await self._execute_compliance_check(rule, trade_data, portfolio_context)

                if check_result.get("violation"):
                    violations.append(f"{rule.rule_name}: {check_result['message']}")
                elif check_result.get("warning"):
                    warnings.append(f"{rule.rule_name}: {check_result['message']}")

            except Exception as e:
                logger.error(f"Compliance check failed for rule {rule.rule_id}: {e}")
                warnings.append(f"{rule.rule_name}: Check failed - {str(e)}")

        # Determine overall status
        if violations:
            status = ComplianceStatus.VIOLATION
        elif warnings:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.COMPLIANT

        result = ComplianceCheckResult(
            check_id=check_id,
            timestamp=timestamp,
            rules_checked=rules_checked,
            violations=violations,
            warnings=warnings,
            status=status,
            remediation_required=len(violations) > 0
        )

        # Store result
        self.compliance_checks[check_id] = result

        # Log audit event
        await self._log_audit_event(
            AuditEventType.COMPLIANCE_CHECK,
            f"Compliance check completed: {status.value}",
            {"check_id": check_id, "violations": len(violations), "warnings": len(warnings)},
            compliance_impact=status
        )

        return result

    async def _execute_compliance_check(
        self,
        rule: ComplianceRule,
        trade_data: Dict[str, Any],
        portfolio_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a specific compliance check."""
        if rule.check_function == "check_position_limits":
            return await self._check_position_limits(rule, trade_data, portfolio_context)
        elif rule.check_function == "check_leverage_limits":
            return await self._check_leverage_limits(rule, trade_data, portfolio_context)
        elif rule.check_function == "check_wash_sales":
            return await self._check_wash_sales(rule, trade_data, portfolio_context)
        elif rule.check_function == "check_best_execution":
            return await self._check_best_execution(rule, trade_data, portfolio_context)
        elif rule.check_function == "check_market_abuse":
            return await self._check_market_abuse(rule, trade_data, portfolio_context)
        else:
            return {"warning": True, "message": f"Unknown check function: {rule.check_function}"}

    async def _check_position_limits(self, rule: ComplianceRule, trade_data: Dict[str, Any], portfolio_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check position size limits."""
        max_position_pct = rule.parameters.get("max_position_pct", 0.10)

        # Simplified check - would use real portfolio data
        trade_size = trade_data.get("quantity", 0) * trade_data.get("price", 0)
        portfolio_value = portfolio_context.get("total_value", 1000000) if portfolio_context else 1000000

        position_pct = trade_size / portfolio_value

        if position_pct > max_position_pct:
            return {"violation": True, "message": f"Position size {position_pct:.2%} exceeds limit {max_position_pct:.2%}"}
        elif position_pct > max_position_pct * 0.8:
            return {"warning": True, "message": f"Position size {position_pct:.2%} approaching limit {max_position_pct:.2%}"}
        else:
            return {"compliant": True}

    async def _check_leverage_limits(self, rule: ComplianceRule, trade_data: Dict[str, Any], portfolio_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check leverage limits."""
        max_leverage = rule.parameters.get("max_leverage", 2.0)

        # Simplified leverage calculation
        current_leverage = portfolio_context.get("leverage", 1.0) if portfolio_context else 1.0

        if current_leverage > max_leverage:
            return {"violation": True, "message": f"Leverage {current_leverage:.2f} exceeds limit {max_leverage:.2f}"}
        elif current_leverage > max_leverage * 0.9:
            return {"warning": True, "message": f"Leverage {current_leverage:.2f} approaching limit {max_leverage:.2f}"}
        else:
            return {"compliant": True}

    async def _check_wash_sales(self, rule: ComplianceRule, trade_data: Dict[str, Any], portfolio_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check for wash sale violations."""
        # Simplified wash sale check
        symbol = trade_data.get("symbol")
        side = trade_data.get("side")

        # Would check trade history for same symbol within lookback period
        # For demo, assume compliant
        return {"compliant": True}

    async def _check_best_execution(self, rule: ComplianceRule, trade_data: Dict[str, Any], portfolio_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check best execution requirements."""
        # Simplified best execution check
        execution_price = trade_data.get("price", 0)
        market_price = trade_data.get("market_price", execution_price)

        price_diff = abs(execution_price - market_price) / market_price if market_price else 0
        threshold = rule.parameters.get("price_improvement_threshold", 0.001)

        if price_diff > threshold:
            return {"warning": True, "message": f"Execution price deviation {price_diff:.4f} exceeds threshold {threshold:.4f}"}
        else:
            return {"compliant": True}

    async def _check_market_abuse(self, rule: ComplianceRule, trade_data: Dict[str, Any], portfolio_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check for potential market abuse patterns."""
        # Simplified market abuse detection
        trade_volume = trade_data.get("quantity", 0)
        market_volume = trade_data.get("market_volume", trade_volume * 100)

        volume_pct = trade_volume / market_volume if market_volume else 0
        threshold = rule.parameters.get("volume_threshold", 0.05)

        if volume_pct > threshold:
            return {"warning": True, "message": f"Trade volume {volume_pct:.2%} of market volume may impact price"}
        else:
            return {"compliant": True}

    async def _log_audit_event(
        self,
        event_type: AuditEventType,
        description: str,
        details: Dict[str, Any] = None,
        user_id: str = "system",
        risk_impact: Optional[RiskLevel] = None,
        compliance_impact: Optional[ComplianceStatus] = None
    ) -> None:
        """Log an audit event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            user_id=user_id,
            description=description,
            details=details or {},
            risk_impact=risk_impact,
            compliance_impact=compliance_impact
        )

        self.audit_events.append(event)

        # Write to audit file
        try:
            with open(self.audit_file, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit event to file: {e}")

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary."""
        recent_assessments = [
            assessment for assessment in self.risk_assessments.values()
            if assessment.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
        ]

        return {
            "total_assessments": len(self.risk_assessments),
            "recent_assessments": len(recent_assessments),
            "current_limits": self.risk_limits.to_dict(),
            "breaches_today": sum(1 for a in recent_assessments if a.limit_breaches),
            "high_risk_assessments": sum(1 for a in recent_assessments if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
        }

    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary."""
        recent_checks = [
            check for check in self.compliance_checks.values()
            if check.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
        ]

        return {
            "total_checks": len(self.compliance_checks),
            "recent_checks": len(recent_checks),
            "active_rules": len([r for r in self.compliance_rules if r.enabled]),
            "violations_today": sum(1 for c in recent_checks if c.violations),
            "remediation_required": sum(1 for c in recent_checks if c.remediation_required)
        }

    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit trail summary."""
        recent_events = [
            event for event in self.audit_events
            if event.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
        ]

        return {
            "total_events": len(self.audit_events),
            "recent_events": len(recent_events),
            "audit_file": str(self.audit_file),
            "event_types": {event_type.value: sum(1 for e in recent_events if e.event_type == event_type) for event_type in AuditEventType}
        }


__all__ = [
    "RiskCompliancePipeline",
    "RiskMetrics",
    "RiskLimits",
    "ComplianceRule",
    "AuditEvent",
    "RiskAssessmentResult",
    "ComplianceCheckResult",
    "RiskLevel",
    "ComplianceStatus",
    "AuditEventType"
]