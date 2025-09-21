import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class ComplianceRuleType(Enum):
    POSITION_LIMIT = "position_limit"
    SECTOR_CONCENTRATION = "sector_concentration"
    LIQUIDITY_REQUIREMENT = "liquidity_requirement"
    VOLATILITY_LIMIT = "volatility_limit"
    LEVERAGE_CONSTRAINT = "leverage_constraint"
    RISK_LIMIT = "risk_limit"
    TRADING_RESTRICTION = "trading_restriction"
    REPORTING_REQUIREMENT = "reporting_requirement"

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    UNDER_REVIEW = "under_review"

class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ComplianceRule:
    """Individual compliance rule definition"""
    rule_id: str
    name: str
    rule_type: ComplianceRuleType
    description: str
    parameters: Dict[str, Any]
    severity: Severity
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    regulation_reference: Optional[str] = None
    jurisdiction: str = "US"

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    rule_id: str
    rule_name: str
    severity: Severity
    status: ComplianceStatus
    description: str
    detected_at: datetime
    portfolio_snapshot: Dict[str, Any]
    remediation_actions: List[str] = field(default_factory=list)
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None

@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    report_id: str
    report_date: datetime
    portfolio_value: float
    total_violations: int
    violations_by_severity: Dict[Severity, int]
    rule_compliance_rates: Dict[str, float]
    violations: List[ComplianceViolation]
    recommendations: List[str]
    next_review_date: datetime

class ComplianceRuleEngine:
    """Core engine for evaluating compliance rules"""

    def __init__(self):
        self.rules: Dict[str, ComplianceRule] = {}
        self._rule_evaluators: Dict[ComplianceRuleType, Callable] = {
            ComplianceRuleType.POSITION_LIMIT: self._evaluate_position_limits,
            ComplianceRuleType.SECTOR_CONCENTRATION: self._evaluate_sector_concentration,
            ComplianceRuleType.LIQUIDITY_REQUIREMENT: self._evaluate_liquidity_requirements,
            ComplianceRuleType.VOLATILITY_LIMIT: self._evaluate_volatility_limits,
            ComplianceRuleType.LEVERAGE_CONSTRAINT: self._evaluate_leverage_constraints,
            ComplianceRuleType.RISK_LIMIT: self._evaluate_risk_limits,
            ComplianceRuleType.TRADING_RESTRICTION: self._evaluate_trading_restrictions,
            ComplianceRuleType.REPORTING_REQUIREMENT: self._evaluate_reporting_requirements
        }

    def add_rule(self, rule: ComplianceRule) -> bool:
        """Add a new compliance rule"""
        try:
            self.rules[rule.rule_id] = rule
            logger.info(f"Added compliance rule: {rule.name} ({rule.rule_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to add rule {rule.rule_id}: {e}")
            return False

    def evaluate_portfolio(
        self,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        transaction_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[ComplianceViolation]:
        """Evaluate portfolio against all active compliance rules"""
        violations = []

        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue

            try:
                evaluator = self._rule_evaluators.get(rule.rule_type)
                if evaluator:
                    violation = evaluator(rule, portfolio, market_data, transaction_history)
                    if violation:
                        violations.append(violation)
                else:
                    logger.warning(f"No evaluator found for rule type: {rule.rule_type}")

            except Exception as e:
                logger.error(f"Error evaluating rule {rule_id}: {e}")
                # Create a system error violation
                error_violation = ComplianceViolation(
                    violation_id=f"error_{rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    rule_id=rule_id,
                    rule_name=rule.name,
                    severity=Severity.MEDIUM,
                    status=ComplianceStatus.UNDER_REVIEW,
                    description=f"System error during rule evaluation: {str(e)}",
                    detected_at=datetime.now(),
                    portfolio_snapshot=portfolio
                )
                violations.append(error_violation)

        return violations

    def _evaluate_position_limits(
        self,
        rule: ComplianceRule,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        transaction_history: Optional[List[Dict[str, Any]]]
    ) -> Optional[ComplianceViolation]:
        """Evaluate position size limits"""
        max_position_pct = rule.parameters.get('max_position_percentage', 10.0)
        portfolio_value = portfolio.get('total_value', 0)

        if portfolio_value == 0:
            return None

        positions = portfolio.get('positions', {})

        for symbol, position_data in positions.items():
            position_value = abs(position_data.get('market_value', 0))
            position_pct = (position_value / portfolio_value) * 100

            if position_pct > max_position_pct:
                return ComplianceViolation(
                    violation_id=f"pos_limit_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    status=ComplianceStatus.VIOLATION,
                    description=f"Position {symbol} ({position_pct:.2f}%) exceeds limit ({max_position_pct}%)",
                    detected_at=datetime.now(),
                    portfolio_snapshot=portfolio,
                    remediation_actions=[f"Reduce {symbol} position to comply with {max_position_pct}% limit"]
                )

        return None

    def _evaluate_sector_concentration(
        self,
        rule: ComplianceRule,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        transaction_history: Optional[List[Dict[str, Any]]]
    ) -> Optional[ComplianceViolation]:
        """Evaluate sector concentration limits"""
        max_sector_pct = rule.parameters.get('max_sector_percentage', 25.0)
        portfolio_value = portfolio.get('total_value', 0)

        if portfolio_value == 0:
            return None

        # Aggregate positions by sector
        sector_exposures = {}
        positions = portfolio.get('positions', {})

        for symbol, position_data in positions.items():
            sector = market_data.get(symbol, {}).get('sector', 'Unknown')
            position_value = abs(position_data.get('market_value', 0))

            if sector not in sector_exposures:
                sector_exposures[sector] = 0
            sector_exposures[sector] += position_value

        # Check sector limits
        for sector, exposure in sector_exposures.items():
            sector_pct = (exposure / portfolio_value) * 100

            if sector_pct > max_sector_pct:
                return ComplianceViolation(
                    violation_id=f"sector_conc_{sector}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    status=ComplianceStatus.VIOLATION,
                    description=f"Sector {sector} concentration ({sector_pct:.2f}%) exceeds limit ({max_sector_pct}%)",
                    detected_at=datetime.now(),
                    portfolio_snapshot=portfolio,
                    remediation_actions=[f"Reduce exposure to {sector} sector to comply with {max_sector_pct}% limit"]
                )

        return None

    def _evaluate_liquidity_requirements(
        self,
        rule: ComplianceRule,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        transaction_history: Optional[List[Dict[str, Any]]]
    ) -> Optional[ComplianceViolation]:
        """Evaluate minimum liquidity requirements"""
        min_liquidity_pct = rule.parameters.get('min_liquidity_percentage', 5.0)
        portfolio_value = portfolio.get('total_value', 0)

        if portfolio_value == 0:
            return None

        # Calculate liquid assets (cash + highly liquid securities)
        cash_value = portfolio.get('cash', 0)
        liquid_securities_value = 0

        positions = portfolio.get('positions', {})
        for symbol, position_data in positions.items():
            # Consider positions with high daily volume as liquid
            avg_volume = market_data.get(symbol, {}).get('avg_daily_volume', 0)
            position_value = abs(position_data.get('market_value', 0))

            # Heuristic: if position < 1% of average daily volume, consider liquid
            if avg_volume > 0 and (position_value / avg_volume) < 0.01:
                liquid_securities_value += position_value

        total_liquidity = cash_value + liquid_securities_value
        liquidity_pct = (total_liquidity / portfolio_value) * 100

        if liquidity_pct < min_liquidity_pct:
            return ComplianceViolation(
                violation_id=f"liquidity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                status=ComplianceStatus.VIOLATION,
                description=f"Portfolio liquidity ({liquidity_pct:.2f}%) below required minimum ({min_liquidity_pct}%)",
                detected_at=datetime.now(),
                portfolio_snapshot=portfolio,
                remediation_actions=[
                    f"Increase cash position or liquid securities to meet {min_liquidity_pct}% requirement",
                    "Consider selling less liquid positions"
                ]
            )

        return None

    def _evaluate_volatility_limits(
        self,
        rule: ComplianceRule,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        transaction_history: Optional[List[Dict[str, Any]]]
    ) -> Optional[ComplianceViolation]:
        """Evaluate portfolio volatility limits"""
        max_portfolio_vol = rule.parameters.get('max_portfolio_volatility', 20.0)

        # Calculate portfolio volatility
        positions = portfolio.get('positions', {})
        if not positions:
            return None

        # Simplified volatility calculation (would need correlation matrix for full calculation)
        weighted_vol = 0
        portfolio_value = portfolio.get('total_value', 0)

        for symbol, position_data in positions.items():
            position_value = abs(position_data.get('market_value', 0))
            weight = position_value / portfolio_value if portfolio_value > 0 else 0
            volatility = market_data.get(symbol, {}).get('volatility', 0)

            weighted_vol += weight * (volatility ** 2)

        portfolio_vol = np.sqrt(weighted_vol) * 100  # Convert to percentage

        if portfolio_vol > max_portfolio_vol:
            return ComplianceViolation(
                violation_id=f"volatility_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                status=ComplianceStatus.VIOLATION,
                description=f"Portfolio volatility ({portfolio_vol:.2f}%) exceeds limit ({max_portfolio_vol}%)",
                detected_at=datetime.now(),
                portfolio_snapshot=portfolio,
                remediation_actions=[
                    "Reduce positions in high-volatility assets",
                    "Increase diversification to reduce overall portfolio risk"
                ]
            )

        return None

    def _evaluate_leverage_constraints(
        self,
        rule: ComplianceRule,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        transaction_history: Optional[List[Dict[str, Any]]]
    ) -> Optional[ComplianceViolation]:
        """Evaluate leverage constraints"""
        max_leverage = rule.parameters.get('max_leverage_ratio', 2.0)

        gross_exposure = portfolio.get('gross_exposure', 0)
        net_worth = portfolio.get('net_worth', 0)

        if net_worth == 0:
            return None

        leverage_ratio = gross_exposure / net_worth

        if leverage_ratio > max_leverage:
            return ComplianceViolation(
                violation_id=f"leverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                status=ComplianceStatus.VIOLATION,
                description=f"Leverage ratio ({leverage_ratio:.2f}) exceeds limit ({max_leverage})",
                detected_at=datetime.now(),
                portfolio_snapshot=portfolio,
                remediation_actions=[
                    "Reduce gross exposure by closing positions",
                    "Add capital to increase net worth"
                ]
            )

        return None

    def _evaluate_risk_limits(
        self,
        rule: ComplianceRule,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        transaction_history: Optional[List[Dict[str, Any]]]
    ) -> Optional[ComplianceViolation]:
        """Evaluate Value-at-Risk and other risk limits"""
        max_var = rule.parameters.get('max_var_percentage', 5.0)
        confidence_level = rule.parameters.get('confidence_level', 0.95)

        # Get VaR from portfolio risk metrics
        var_pct = portfolio.get('risk_metrics', {}).get('var_95', 0)

        if var_pct > max_var:
            return ComplianceViolation(
                violation_id=f"var_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                status=ComplianceStatus.VIOLATION,
                description=f"Value-at-Risk ({var_pct:.2f}%) exceeds limit ({max_var}%) at {confidence_level*100}% confidence",
                detected_at=datetime.now(),
                portfolio_snapshot=portfolio,
                remediation_actions=[
                    "Reduce position sizes to lower portfolio risk",
                    "Hedge positions to reduce downside exposure"
                ]
            )

        return None

    def _evaluate_trading_restrictions(
        self,
        rule: ComplianceRule,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        transaction_history: Optional[List[Dict[str, Any]]]
    ) -> Optional[ComplianceViolation]:
        """Evaluate trading restrictions (wash sales, pattern day trading, etc.)"""
        if not transaction_history:
            return None

        # Check for wash sale violations
        wash_sale_period = rule.parameters.get('wash_sale_period_days', 30)
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=wash_sale_period)

        recent_transactions = [
            tx for tx in transaction_history
            if datetime.fromisoformat(tx.get('timestamp', '')) >= cutoff_time
        ]

        # Group transactions by symbol
        symbol_transactions = {}
        for tx in recent_transactions:
            symbol = tx.get('symbol')
            if symbol not in symbol_transactions:
                symbol_transactions[symbol] = []
            symbol_transactions[symbol].append(tx)

        # Check for wash sales (buy and sell of same security within period)
        for symbol, transactions in symbol_transactions.items():
            buys = [tx for tx in transactions if tx.get('side') == 'buy']
            sells = [tx for tx in transactions if tx.get('side') == 'sell']

            if buys and sells:
                # Potential wash sale - simplified check
                return ComplianceViolation(
                    violation_id=f"wash_sale_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    status=ComplianceStatus.WARNING,
                    description=f"Potential wash sale detected for {symbol} within {wash_sale_period} days",
                    detected_at=datetime.now(),
                    portfolio_snapshot=portfolio,
                    remediation_actions=[
                        "Review transactions for wash sale compliance",
                        "Consult tax advisor if necessary"
                    ]
                )

        return None

    def _evaluate_reporting_requirements(
        self,
        rule: ComplianceRule,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        transaction_history: Optional[List[Dict[str, Any]]]
    ) -> Optional[ComplianceViolation]:
        """Evaluate reporting requirements (13F, etc.)"""
        reporting_threshold = rule.parameters.get('reporting_threshold', 100000000)  # $100M
        portfolio_value = portfolio.get('total_value', 0)

        if portfolio_value > reporting_threshold:
            # Check if recent report filed
            last_report_date = portfolio.get('last_regulatory_report_date')
            if last_report_date:
                last_report = datetime.fromisoformat(last_report_date)
                days_since_report = (datetime.now() - last_report).days
                max_days = rule.parameters.get('max_days_since_report', 45)

                if days_since_report > max_days:
                    return ComplianceViolation(
                        violation_id=f"reporting_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        status=ComplianceStatus.VIOLATION,
                        description=f"Regulatory report overdue by {days_since_report - max_days} days",
                        detected_at=datetime.now(),
                        portfolio_snapshot=portfolio,
                        remediation_actions=[
                            "File required regulatory report immediately",
                            "Review reporting calendar and procedures"
                        ]
                    )

        return None

class ComplianceAutomationSystem:
    """Automated compliance monitoring and reporting system"""

    def __init__(self, config_path: Optional[str] = None):
        self.rule_engine = ComplianceRuleEngine()
        self.violations_history: List[ComplianceViolation] = []
        self.reports_history: List[ComplianceReport] = []
        self.config_path = config_path
        self._load_configuration()

    def _load_configuration(self):
        """Load compliance rules from configuration"""
        if not self.config_path or not Path(self.config_path).exists():
            self._load_default_rules()
            return

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            for rule_data in config.get('rules', []):
                rule = ComplianceRule(
                    rule_id=rule_data['rule_id'],
                    name=rule_data['name'],
                    rule_type=ComplianceRuleType(rule_data['rule_type']),
                    description=rule_data['description'],
                    parameters=rule_data['parameters'],
                    severity=Severity(rule_data['severity']),
                    enabled=rule_data.get('enabled', True),
                    regulation_reference=rule_data.get('regulation_reference'),
                    jurisdiction=rule_data.get('jurisdiction', 'US')
                )
                self.rule_engine.add_rule(rule)

            logger.info(f"Loaded {len(config.get('rules', []))} compliance rules from configuration")

        except Exception as e:
            logger.error(f"Failed to load compliance configuration: {e}")
            self._load_default_rules()

    def _load_default_rules(self):
        """Load default compliance rules"""
        default_rules = [
            ComplianceRule(
                rule_id="pos_limit_001",
                name="Maximum Position Size",
                rule_type=ComplianceRuleType.POSITION_LIMIT,
                description="No single position can exceed 10% of portfolio value",
                parameters={'max_position_percentage': 10.0},
                severity=Severity.HIGH,
                regulation_reference="Internal Risk Policy"
            ),
            ComplianceRule(
                rule_id="sector_conc_001",
                name="Sector Concentration Limit",
                rule_type=ComplianceRuleType.SECTOR_CONCENTRATION,
                description="No sector can exceed 25% of portfolio value",
                parameters={'max_sector_percentage': 25.0},
                severity=Severity.MEDIUM,
                regulation_reference="Diversification Requirements"
            ),
            ComplianceRule(
                rule_id="liquidity_001",
                name="Minimum Liquidity Requirement",
                rule_type=ComplianceRuleType.LIQUIDITY_REQUIREMENT,
                description="Portfolio must maintain minimum 5% liquidity",
                parameters={'min_liquidity_percentage': 5.0},
                severity=Severity.MEDIUM
            ),
            ComplianceRule(
                rule_id="leverage_001",
                name="Maximum Leverage",
                rule_type=ComplianceRuleType.LEVERAGE_CONSTRAINT,
                description="Leverage ratio cannot exceed 2:1",
                parameters={'max_leverage_ratio': 2.0},
                severity=Severity.CRITICAL,
                regulation_reference="Regulation T"
            ),
            ComplianceRule(
                rule_id="var_001",
                name="Value-at-Risk Limit",
                rule_type=ComplianceRuleType.RISK_LIMIT,
                description="Portfolio VaR cannot exceed 5% at 95% confidence",
                parameters={'max_var_percentage': 5.0, 'confidence_level': 0.95},
                severity=Severity.HIGH
            )
        ]

        for rule in default_rules:
            self.rule_engine.add_rule(rule)

        logger.info(f"Loaded {len(default_rules)} default compliance rules")

    async def monitor_compliance(
        self,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        transaction_history: Optional[List[Dict[str, Any]]] = None
    ) -> ComplianceReport:
        """Perform real-time compliance monitoring"""
        logger.info("Starting compliance monitoring")

        # Evaluate all rules
        violations = self.rule_engine.evaluate_portfolio(portfolio, market_data, transaction_history)

        # Store violations
        self.violations_history.extend(violations)

        # Generate compliance report
        report = self._generate_compliance_report(violations, portfolio)

        # Store report
        self.reports_history.append(report)

        # Take automated actions for critical violations
        await self._handle_critical_violations(violations)

        logger.info(f"Compliance monitoring completed. Found {len(violations)} violations")
        return report

    def _generate_compliance_report(self, violations: List[ComplianceViolation], portfolio: Dict[str, Any]) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        report_id = f"comp_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Count violations by severity
        violations_by_severity = {severity: 0 for severity in Severity}
        for violation in violations:
            violations_by_severity[violation.severity] += 1

        # Calculate rule compliance rates
        total_rules = len(self.rule_engine.rules)
        violated_rules = len(set(v.rule_id for v in violations))
        overall_compliance_rate = ((total_rules - violated_rules) / total_rules * 100) if total_rules > 0 else 100

        rule_compliance_rates = {}
        for rule_id, rule in self.rule_engine.rules.items():
            rule_violations = [v for v in violations if v.rule_id == rule_id]
            rule_compliance_rates[rule_id] = 0.0 if rule_violations else 100.0

        # Generate recommendations
        recommendations = self._generate_recommendations(violations)

        # Set next review date (daily for now)
        next_review_date = datetime.now() + timedelta(days=1)

        return ComplianceReport(
            report_id=report_id,
            report_date=datetime.now(),
            portfolio_value=portfolio.get('total_value', 0),
            total_violations=len(violations),
            violations_by_severity=violations_by_severity,
            rule_compliance_rates=rule_compliance_rates,
            violations=violations,
            recommendations=recommendations,
            next_review_date=next_review_date
        )

    def _generate_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate automated recommendations based on violations"""
        recommendations = []

        # Aggregate recommendations from violations
        all_actions = []
        for violation in violations:
            all_actions.extend(violation.remediation_actions)

        # Remove duplicates and prioritize
        unique_actions = list(set(all_actions))

        # Add general recommendations based on violation patterns
        critical_violations = [v for v in violations if v.severity == Severity.CRITICAL]
        if critical_violations:
            recommendations.append("URGENT: Address critical compliance violations immediately")

        high_violations = [v for v in violations if v.severity == Severity.HIGH]
        if len(high_violations) > 3:
            recommendations.append("Multiple high-severity violations detected - comprehensive portfolio review recommended")

        # Add specific recommendations
        recommendations.extend(unique_actions[:10])  # Limit to top 10

        return recommendations

    async def _handle_critical_violations(self, violations: List[ComplianceViolation]):
        """Handle critical violations with automated actions"""
        critical_violations = [v for v in violations if v.severity == Severity.CRITICAL]

        for violation in critical_violations:
            logger.critical(f"CRITICAL COMPLIANCE VIOLATION: {violation.description}")

            # Send alerts (implementation depends on alerting system)
            await self._send_compliance_alert(violation)

            # Log for audit trail
            self._log_compliance_action(violation, "CRITICAL_ALERT_SENT")

    async def _send_compliance_alert(self, violation: ComplianceViolation):
        """Send compliance alert (placeholder for actual implementation)"""
        logger.warning(f"Compliance alert would be sent for violation: {violation.violation_id}")
        # Implementation would integrate with actual alerting system
        # (email, Slack, PagerDuty, etc.)

    def _log_compliance_action(self, violation: ComplianceViolation, action: str):
        """Log compliance action for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'violation_id': violation.violation_id,
            'action': action,
            'rule_id': violation.rule_id,
            'severity': violation.severity.name
        }
        logger.info(f"Compliance action logged: {json.dumps(log_entry)}")

    def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for compliance dashboard"""
        recent_violations = [
            v for v in self.violations_history
            if v.detected_at >= datetime.now() - timedelta(days=30)
        ]

        return {
            'total_rules': len(self.rule_engine.rules),
            'active_rules': len([r for r in self.rule_engine.rules.values() if r.enabled]),
            'recent_violations': len(recent_violations),
            'critical_violations': len([v for v in recent_violations if v.severity == Severity.CRITICAL]),
            'violation_trend': self._calculate_violation_trend(),
            'compliance_score': self._calculate_compliance_score(),
            'top_violated_rules': self._get_top_violated_rules()
        }

    def _calculate_violation_trend(self) -> List[Dict[str, Any]]:
        """Calculate violation trend over time"""
        # Group violations by day for the last 30 days
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)

        daily_violations = {}
        current_date = start_date

        while current_date <= end_date:
            daily_violations[current_date] = 0
            current_date += timedelta(days=1)

        for violation in self.violations_history:
            violation_date = violation.detected_at.date()
            if start_date <= violation_date <= end_date:
                daily_violations[violation_date] += 1

        return [
            {'date': date.isoformat(), 'violations': count}
            for date, count in sorted(daily_violations.items())
        ]

    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)"""
        if not self.violations_history:
            return 100.0

        # Weight violations by severity and recency
        recent_violations = [
            v for v in self.violations_history
            if v.detected_at >= datetime.now() - timedelta(days=7)
        ]

        if not recent_violations:
            return 95.0  # Good score if no recent violations

        severity_weights = {
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 4,
            Severity.CRITICAL: 8
        }

        total_weight = sum(severity_weights[v.severity] for v in recent_violations)
        max_possible_weight = len(self.rule_engine.rules) * severity_weights[Severity.CRITICAL]

        if max_possible_weight == 0:
            return 100.0

        compliance_score = max(0, 100 - (total_weight / max_possible_weight * 100))
        return round(compliance_score, 2)

    def _get_top_violated_rules(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most frequently violated rules"""
        rule_violation_counts = {}

        for violation in self.violations_history:
            rule_id = violation.rule_id
            if rule_id not in rule_violation_counts:
                rule_violation_counts[rule_id] = {
                    'rule_id': rule_id,
                    'rule_name': violation.rule_name,
                    'count': 0
                }
            rule_violation_counts[rule_id]['count'] += 1

        # Sort by count and return top N
        sorted_rules = sorted(rule_violation_counts.values(), key=lambda x: x['count'], reverse=True)
        return sorted_rules[:limit]

# Utility functions for integration
def create_compliance_automation() -> ComplianceAutomationSystem:
    """Create a compliance automation system with default configuration"""
    return ComplianceAutomationSystem()

async def run_compliance_check(
    portfolio: Dict[str, Any],
    market_data: Dict[str, Any],
    transaction_history: Optional[List[Dict[str, Any]]] = None
) -> ComplianceReport:
    """Convenience function to run a compliance check"""
    compliance_system = create_compliance_automation()
    return await compliance_system.monitor_compliance(portfolio, market_data, transaction_history)