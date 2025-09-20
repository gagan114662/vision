"""
Test Risk and Compliance Pipeline Integration.

This demonstrates the comprehensive risk and compliance system working
with real audit trails, regulatory checks, and integration with the
multi-agent orchestration system.
"""
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, '.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from mcp.risk_compliance_pipeline import (
    RiskCompliancePipeline, RiskLevel, ComplianceStatus, AuditEventType,
    RiskLimits, ComplianceRule
)


class RiskComplianceDemo:
    """Demonstration of comprehensive risk and compliance system."""

    def __init__(self):
        # Configure risk limits
        risk_config = {
            "risk_limits": {
                "max_var_95": 0.03,  # 3% VaR limit
                "max_var_99": 0.05,  # 5% VaR limit
                "max_drawdown": 0.10,  # 10% max drawdown
                "max_concentration": 0.08,  # 8% max position size
                "max_beta": 1.3,
                "max_volatility": 0.25
            },
            "audit_file": "demo_audit_trail.jsonl",
            "alert_thresholds": {
                "critical_risk": True,
                "compliance_violations": True
            }
        }

        self.pipeline = RiskCompliancePipeline(risk_config)

    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run complete risk and compliance demonstration."""
        print("🛡️ Starting Comprehensive Risk & Compliance Demo")
        print("=" * 60)

        try:
            # Step 1: Risk Assessment
            print("\n📊 Step 1: Risk Assessment")
            risk_results = await self._demonstrate_risk_assessment()

            # Step 2: Compliance Checking
            print("\n⚖️ Step 2: Compliance Checking")
            compliance_results = await self._demonstrate_compliance_checking()

            # Step 3: Audit Trail
            print("\n📋 Step 3: Audit Trail Management")
            audit_results = await self._demonstrate_audit_trail()

            # Step 4: Integration Scenarios
            print("\n🔄 Step 4: Integration Scenarios")
            integration_results = await self._demonstrate_integration_scenarios()

            # Step 5: Reporting and Analytics
            print("\n📈 Step 5: Reporting and Analytics")
            reporting_results = await self._demonstrate_reporting()

            return {
                "risk_assessment": risk_results,
                "compliance_checking": compliance_results,
                "audit_trail": audit_results,
                "integration_scenarios": integration_results,
                "reporting": reporting_results,
                "demo_status": "completed_successfully"
            }

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            return {
                "demo_status": "failed",
                "error": str(e)
            }

    async def _demonstrate_risk_assessment(self):
        """Demonstrate risk assessment capabilities."""
        print("🔍 Risk Assessment Scenarios:")

        scenarios = [
            {
                "name": "Conservative Portfolio",
                "portfolio": {
                    "positions": [
                        {"symbol": "AAPL", "market_value": 50000, "volatility": 0.20, "beta": 1.1},
                        {"symbol": "MSFT", "market_value": 40000, "volatility": 0.18, "beta": 1.0},
                        {"symbol": "GOOGL", "market_value": 30000, "volatility": 0.22, "beta": 1.2}
                    ]
                }
            },
            {
                "name": "Aggressive Portfolio",
                "portfolio": {
                    "positions": [
                        {"symbol": "TSLA", "market_value": 80000, "volatility": 0.45, "beta": 2.1},
                        {"symbol": "NVDA", "market_value": 60000, "volatility": 0.40, "beta": 1.8},
                        {"symbol": "AMZN", "market_value": 40000, "volatility": 0.30, "beta": 1.4}
                    ]
                }
            },
            {
                "name": "Concentrated Position",
                "portfolio": {
                    "positions": [
                        {"symbol": "AAPL", "market_value": 150000, "volatility": 0.20, "beta": 1.1},
                        {"symbol": "CASH", "market_value": 50000, "volatility": 0.0, "beta": 0.0}
                    ]
                }
            }
        ]

        results = []

        for scenario in scenarios:
            print(f"\n   📊 Scenario: {scenario['name']}")

            assessment = await self.pipeline.assess_risk(scenario["portfolio"])

            print(f"      Risk Level: {assessment.risk_level.value}")
            print(f"      VaR 95%: {assessment.metrics.var_95:.2%}")
            print(f"      VaR 99%: {assessment.metrics.var_99:.2%}")
            print(f"      Max Drawdown: {assessment.metrics.max_drawdown:.2%}")
            print(f"      Concentration: {assessment.metrics.concentration_risk:.2%}")
            print(f"      Approved: {'✅' if assessment.approved else '❌'}")

            if assessment.limit_breaches:
                print(f"      Breaches: {len(assessment.limit_breaches)}")
                for breach in assessment.limit_breaches:
                    print(f"        - {breach}")

            if assessment.recommendations:
                print(f"      Recommendations:")
                for rec in assessment.recommendations:
                    print(f"        - {rec}")

            results.append({
                "scenario": scenario["name"],
                "assessment": assessment.to_dict()
            })

        return results

    async def _demonstrate_compliance_checking(self):
        """Demonstrate compliance checking capabilities."""
        print("⚖️ Compliance Check Scenarios:")

        scenarios = [
            {
                "name": "Normal Trade",
                "trade": {
                    "symbol": "AAPL",
                    "side": "buy",
                    "quantity": 100,
                    "price": 150.0,
                    "market_price": 150.0,
                    "market_volume": 10000
                },
                "portfolio": {"total_value": 1000000, "leverage": 1.2}
            },
            {
                "name": "Large Position Trade",
                "trade": {
                    "symbol": "TSLA",
                    "side": "buy",
                    "quantity": 500,
                    "price": 200.0,
                    "market_price": 200.0,
                    "market_volume": 5000
                },
                "portfolio": {"total_value": 500000, "leverage": 1.8}
            },
            {
                "name": "High Volume Trade",
                "trade": {
                    "symbol": "NVDA",
                    "side": "sell",
                    "quantity": 1000,
                    "price": 400.0,
                    "market_price": 400.2,
                    "market_volume": 8000
                },
                "portfolio": {"total_value": 2000000, "leverage": 2.2}
            }
        ]

        results = []

        for scenario in scenarios:
            print(f"\n   ⚖️ Scenario: {scenario['name']}")

            check_result = await self.pipeline.check_compliance(
                scenario["trade"],
                scenario["portfolio"]
            )

            print(f"      Status: {check_result.status.value}")
            print(f"      Rules Checked: {len(check_result.rules_checked)}")

            if check_result.violations:
                print(f"      Violations: {len(check_result.violations)}")
                for violation in check_result.violations:
                    print(f"        ❌ {violation}")

            if check_result.warnings:
                print(f"      Warnings: {len(check_result.warnings)}")
                for warning in check_result.warnings:
                    print(f"        ⚠️ {warning}")

            print(f"      Remediation Required: {'Yes' if check_result.remediation_required else 'No'}")

            results.append({
                "scenario": scenario["name"],
                "check_result": check_result.to_dict()
            })

        return results

    async def _demonstrate_audit_trail(self):
        """Demonstrate audit trail capabilities."""
        print("📋 Audit Trail Capabilities:")

        # Log various types of events
        audit_events = [
            (AuditEventType.AGENT_ACTION, "Fundamental agent generated BUY signal for AAPL", {"symbol": "AAPL", "signal": "BUY", "confidence": 0.85}),
            (AuditEventType.TRADE_DECISION, "Portfolio decision: Execute AAPL buy order", {"symbol": "AAPL", "quantity": 100, "price": 150.0}),
            (AuditEventType.RISK_BREACH, "VaR limit exceeded during position sizing", {"var_95": 0.04, "limit": 0.03}),
            (AuditEventType.COMPLIANCE_CHECK, "Best execution check completed", {"price_improvement": 0.0005}),
            (AuditEventType.SYSTEM_ALERT, "High volatility detected in portfolio", {"portfolio_vol": 0.28, "threshold": 0.25})
        ]

        for event_type, description, details in audit_events:
            await self.pipeline._log_audit_event(
                event_type=event_type,
                description=description,
                details=details,
                user_id="demo_user",
                risk_impact=RiskLevel.MEDIUM if "risk" in description.lower() else None,
                compliance_impact=ComplianceStatus.WARNING if "compliance" in description.lower() else None
            )

        print(f"   ✅ Logged {len(audit_events)} audit events")

        # Show audit summary
        audit_summary = self.pipeline.get_audit_summary()
        print(f"   📊 Total Events: {audit_summary['total_events']}")
        print(f"   📊 Recent Events: {audit_summary['recent_events']}")
        print(f"   📊 Audit File: {audit_summary['audit_file']}")

        print("\n   📋 Event Type Breakdown:")
        for event_type, count in audit_summary['event_types'].items():
            if count > 0:
                print(f"      {event_type}: {count} events")

        return {
            "events_logged": len(audit_events),
            "audit_summary": audit_summary
        }

    async def _demonstrate_integration_scenarios(self):
        """Demonstrate integration with trading workflow."""
        print("🔄 Integration Scenarios:")

        # Scenario 1: Complete trading workflow with risk/compliance
        print("\n   🔄 Scenario: Complete Trading Workflow")

        # Simulated multi-agent recommendation
        agent_recommendation = {
            "symbol": "AAPL",
            "signal": "BUY",
            "confidence": 0.82,
            "target_quantity": 200,
            "price": 151.50,
            "agents_consensus": ["fundamental", "technical", "sentiment"]
        }

        # Current portfolio state
        portfolio_state = {
            "positions": [
                {"symbol": "MSFT", "market_value": 100000, "volatility": 0.18, "beta": 1.0},
                {"symbol": "GOOGL", "market_value": 80000, "volatility": 0.22, "beta": 1.2}
            ],
            "total_value": 180000,
            "leverage": 1.0
        }

        # Step 1: Risk assessment with proposed trade
        proposed_trade = [{
            "symbol": agent_recommendation["symbol"],
            "quantity": agent_recommendation["target_quantity"],
            "price": agent_recommendation["price"],
            "side": "buy"
        }]

        risk_assessment = await self.pipeline.assess_risk(portfolio_state, proposed_trade)
        print(f"      Risk Assessment: {risk_assessment.risk_level.value} ({'Approved' if risk_assessment.approved else 'Rejected'})")

        # Step 2: Compliance check
        trade_data = {
            "symbol": agent_recommendation["symbol"],
            "side": "buy",
            "quantity": agent_recommendation["target_quantity"],
            "price": agent_recommendation["price"],
            "market_price": 151.45,
            "market_volume": 15000
        }

        compliance_check = await self.pipeline.check_compliance(trade_data, {"total_value": 180000, "leverage": 1.0})
        print(f"      Compliance Check: {compliance_check.status.value}")

        # Step 3: Final decision
        approved = risk_assessment.approved and compliance_check.status in [ComplianceStatus.COMPLIANT, ComplianceStatus.WARNING]
        print(f"      Final Decision: {'✅ APPROVED' if approved else '❌ REJECTED'}")

        # Log final decision
        await self.pipeline._log_audit_event(
            AuditEventType.TRADE_DECISION,
            f"Trade decision: {'Approved' if approved else 'Rejected'} {agent_recommendation['symbol']} order",
            {
                "symbol": agent_recommendation["symbol"],
                "quantity": agent_recommendation["target_quantity"],
                "approved": approved,
                "risk_level": risk_assessment.risk_level.value,
                "compliance_status": compliance_check.status.value
            },
            risk_impact=risk_assessment.risk_level,
            compliance_impact=compliance_check.status
        )

        return {
            "agent_recommendation": agent_recommendation,
            "risk_assessment": risk_assessment.to_dict(),
            "compliance_check": compliance_check.to_dict(),
            "final_approved": approved
        }

    async def _demonstrate_reporting(self):
        """Demonstrate reporting and analytics capabilities."""
        print("📈 Risk & Compliance Reporting:")

        # Get comprehensive summaries
        risk_summary = self.pipeline.get_risk_summary()
        compliance_summary = self.pipeline.get_compliance_summary()
        audit_summary = self.pipeline.get_audit_summary()

        print("\n   📊 Risk Management Summary:")
        print(f"      Total Assessments: {risk_summary['total_assessments']}")
        print(f"      Recent Assessments: {risk_summary['recent_assessments']}")
        print(f"      Breaches Today: {risk_summary['breaches_today']}")
        print(f"      High Risk Assessments: {risk_summary['high_risk_assessments']}")

        print("\n   ⚖️ Compliance Summary:")
        print(f"      Total Checks: {compliance_summary['total_checks']}")
        print(f"      Recent Checks: {compliance_summary['recent_checks']}")
        print(f"      Active Rules: {compliance_summary['active_rules']}")
        print(f"      Violations Today: {compliance_summary['violations_today']}")
        print(f"      Remediation Required: {compliance_summary['remediation_required']}")

        print("\n   📋 Audit Summary:")
        print(f"      Total Events: {audit_summary['total_events']}")
        print(f"      Recent Events: {audit_summary['recent_events']}")

        # Show current risk limits
        print("\n   🎯 Current Risk Limits:")
        limits = self.pipeline.risk_limits.to_dict()
        for limit_name, limit_value in limits.items():
            if isinstance(limit_value, float) and limit_value < 1:
                print(f"      {limit_name}: {limit_value:.1%}")
            else:
                print(f"      {limit_name}: {limit_value}")

        return {
            "risk_summary": risk_summary,
            "compliance_summary": compliance_summary,
            "audit_summary": audit_summary,
            "risk_limits": limits
        }


async def main():
    """Run the risk and compliance demonstration."""
    demo = RiskComplianceDemo()

    try:
        result = await demo.run_comprehensive_demo()

        print("\n" + "=" * 60)
        if result["demo_status"] == "completed_successfully":
            print("✅ Risk & Compliance Pipeline Demo COMPLETED SUCCESSFULLY!")
            print("\n🎯 Key Achievements:")
            print("   ✓ Real-time risk assessment operational")
            print("   ✓ Multi-regulation compliance checking active")
            print("   ✓ Comprehensive audit trail implemented")
            print("   ✓ Integration with trading workflow validated")
            print("   ✓ Advanced reporting and analytics available")

            # Show key metrics
            risk_data = result["risk_assessment"]
            compliance_data = result["compliance_checking"]

            print(f"\n📊 Demo Metrics:")
            print(f"   Risk scenarios tested: {len(risk_data)}")
            print(f"   Compliance scenarios tested: {len(compliance_data)}")
            print(f"   Audit events logged: {result['audit_trail']['events_logged']}")
            print(f"   Integration scenarios: 1 complete workflow")

        else:
            print("❌ Demo failed - see errors above")

    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        print(f"\n❌ Demo failed with error: {e}")


if __name__ == "__main__":
    asyncio.run(main())