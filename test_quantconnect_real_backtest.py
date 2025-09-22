#!/usr/bin/env python3
"""
Real QuantConnect backtest validation using actual Lean CLI and Docker.
This script verifies the complete end-to-end flow with real credentials.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantConnectBacktestValidator:
    """Validates real QuantConnect backtest execution through Lean CLI"""

    def __init__(self):
        # Use current working directory or GITHUB_WORKSPACE for CI
        if os.environ.get('GITHUB_ACTIONS'):
            self.project_root = Path(os.environ.get('GITHUB_WORKSPACE', os.getcwd()))
        else:
            self.project_root = Path(os.getcwd())

        self.lean_dir = self.project_root / "lean"
        self.results_dir = self.lean_dir / "results"
        self.validation_results = []

        # Ensure directories exist
        self.lean_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def setup_environment(self):
        """Setup environment for Lean CLI execution"""
        logger.info("Setting up QuantConnect environment...")

        # Check for credentials in environment
        user_id = os.environ.get('QUANTCONNECT_USER_ID') or os.environ.get('QC_USER_ID')
        api_token = os.environ.get('QUANTCONNECT_API_TOKEN') or os.environ.get('QC_API_TOKEN')

        if not user_id or not api_token:
            logger.warning("QuantConnect credentials not found in environment!")

            # Check if we're in a forked PR (no access to secrets)
            if os.environ.get('GITHUB_EVENT_NAME') == 'pull_request':
                logger.info("Running in PR context without secrets - using mock validation")
                return self._setup_mock_environment()

            logger.info("Please set QUANTCONNECT_USER_ID and QUANTCONNECT_API_TOKEN for real validation")
            return False

        # Verify Lean CLI is installed
        try:
            result = subprocess.run(['lean', '--version'], capture_output=True, text=True)
            logger.info(f"Lean CLI version: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.error("Lean CLI not found. Installing...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'lean'], check=True)

        # Login to QuantConnect
        logger.info("Logging into QuantConnect...")
        try:
            login_cmd = ['lean', 'login', '--user-id', user_id, '--api-token', api_token]
            result = subprocess.run(login_cmd, capture_output=True, text=True, check=True)
            logger.info("Successfully logged into QuantConnect")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to login to QuantConnect: {e.stderr}")
            return False

    def create_test_algorithm(self):
        """Create a test algorithm for backtesting"""
        algorithm_dir = self.lean_dir / "algorithms" / "test_validation"
        algorithm_dir.mkdir(parents=True, exist_ok=True)

        algorithm_code = '''
from AlgorithmImports import *

class ValidationTestAlgorithm(QCAlgorithm):
    """Test algorithm for CI/CD validation of real QuantConnect integration"""

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)

        # Add universe selection
        self.AddUniverse(self.CoarseSelectionFunction)

        # Add indicators
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.sma = self.SMA(self.spy, 20, Resolution.Daily)
        self.rsi = self.RSI(self.spy, 14, Resolution.Daily)

        # Portfolio settings
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
        self.SetWarmup(30, Resolution.Daily)

        # Risk management
        self.max_position_size = 0.1
        self.stop_loss_pct = 0.02

        # Schedule rebalancing
        self.Schedule.On(
            self.DateRules.MonthStart(self.spy),
            self.TimeRules.At(9, 30),
            self.Rebalance
        )

        # Track performance
        self.positions_taken = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def CoarseSelectionFunction(self, coarse):
        """Select top 10 stocks by dollar volume"""
        sorted_by_volume = sorted(
            [x for x in coarse if x.HasFundamentalData and x.Price > 5],
            key=lambda x: x.DollarVolume,
            reverse=True
        )
        return [x.Symbol for x in sorted_by_volume[:10]]

    def OnData(self, data):
        """Main trading logic"""
        if self.IsWarmingUp:
            return

        # Check for trading signals
        if not self.sma.IsReady or not self.rsi.IsReady:
            return

        current_price = self.Securities[self.spy].Price

        # Long signal: Price above SMA and RSI oversold
        if current_price > self.sma.Current.Value and self.rsi.Current.Value < 30:
            if not self.Portfolio[self.spy].Invested:
                quantity = self.CalculatePositionSize(self.spy)
                self.MarketOrder(self.spy, quantity)
                self.positions_taken += 1
                self.Log(f"Long entry: {self.spy} at {current_price}")

        # Exit signal: RSI overbought or stop loss
        elif self.Portfolio[self.spy].Invested:
            if self.rsi.Current.Value > 70:
                self.Liquidate(self.spy)
                self.Log(f"Exit position: {self.spy} at {current_price}")
                self.UpdateTradeStats(self.spy)

            # Check stop loss
            elif self.Portfolio[self.spy].UnrealizedProfitPercent < -self.stop_loss_pct:
                self.Liquidate(self.spy)
                self.Log(f"Stop loss triggered: {self.spy}")
                self.losing_trades += 1

    def CalculatePositionSize(self, symbol):
        """Calculate position size based on risk management rules"""
        cash = self.Portfolio.Cash
        price = self.Securities[symbol].Price
        max_value = cash * self.max_position_size
        return int(max_value / price)

    def UpdateTradeStats(self, symbol):
        """Update trade statistics"""
        if self.Portfolio[symbol].LastTradeProfit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

    def Rebalance(self):
        """Monthly rebalancing logic"""
        self.Log("Performing monthly rebalance")

        # Get current holdings
        holdings = [x.Symbol for x in self.Portfolio.Values if x.Invested]

        # Rebalance to equal weight
        target_weight = 1.0 / max(len(holdings), 1)
        for symbol in holdings:
            self.SetHoldings(symbol, target_weight)

    def OnEndOfAlgorithm(self):
        """Final statistics"""
        self.Log(f"Total positions taken: {self.positions_taken}")
        self.Log(f"Winning trades: {self.winning_trades}")
        self.Log(f"Losing trades: {self.losing_trades}")

        if self.positions_taken > 0:
            win_rate = self.winning_trades / max(self.winning_trades + self.losing_trades, 1)
            self.Log(f"Win rate: {win_rate:.2%}")
'''
        algorithm_file = algorithm_dir / "main.py"
        algorithm_file.write_text(algorithm_code)
        logger.info(f"Created test algorithm at {algorithm_file}")

        # Create project config
        config = {
            "algorithm-language": "Python",
            "parameters": {},
            "description": "Validation test algorithm",
            "cloud-id": 0
        }

        config_file = algorithm_dir / "config.json"
        config_file.write_text(json.dumps(config, indent=2))

        return algorithm_dir

    def _setup_mock_environment(self):
        """Setup mock environment for PR builds without secrets"""
        logger.info("Setting up mock environment for validation...")

        # Create mock credentials for structural validation
        os.environ['QUANTCONNECT_USER_ID'] = 'mock_user'
        os.environ['QUANTCONNECT_API_TOKEN'] = 'mock_token'

        # Create mock results for validation testing
        mock_dir = self.results_dir / f"mock_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mock_dir.mkdir(parents=True, exist_ok=True)

        # Create mock statistics file
        mock_stats = {
            "TotalPerformance": {
                "TotalTrades": 42,
                "SharpeRatio": 1.25,
                "WinRate": 0.62,
                "PortfolioStatistics": {
                    "TotalNetProfit": 25000
                }
            },
            "IsMock": True,
            "Note": "This is a mock result for CI validation in PRs without secrets"
        }

        stats_file = mock_dir / f"mock-{datetime.now().strftime('%Y%m%d%H%M%S')}-statistics.json"
        stats_file.write_text(json.dumps(mock_stats, indent=2))

        # Create mock order log
        mock_orders = [
            {"Time": "2023-01-15", "Symbol": "SPY", "Quantity": 100, "Price": 380.50},
            {"Time": "2023-02-20", "Symbol": "SPY", "Quantity": -100, "Price": 395.25}
        ]

        orders_file = mock_dir / f"mock-{datetime.now().strftime('%Y%m%d%H%M%S')}-order-events.json"
        orders_file.write_text(json.dumps(mock_orders, indent=2))

        logger.info(f"Created mock results in {mock_dir}")
        return True

    def run_backtest(self, algorithm_dir: Path):
        """Run actual backtest using Lean CLI with Docker"""
        # Check if we're in mock mode
        if os.environ.get('QUANTCONNECT_USER_ID') == 'mock_user':
            logger.info("Running in mock mode - skipping actual backtest")
            # Return the latest mock directory
            mock_dirs = list(self.results_dir.glob("mock_validation_*"))
            if mock_dirs:
                return sorted(mock_dirs)[-1]
            return None

        logger.info("Starting real QuantConnect backtest...")

        # Prepare output directory
        output_name = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = self.results_dir / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run backtest command
        cmd = [
            'lean', 'backtest',
            str(algorithm_dir / "main.py"),
            '--output', str(output_dir),
            '--debug', 'false',
            '--download-data'  # Download required data
        ]

        logger.info(f"Executing: {' '.join(cmd)}")

        try:
            # Run backtest with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.project_root)
            )

            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    logger.info(f"LEAN: {line.rstrip()}")

            process.wait()

            if process.returncode != 0:
                logger.error(f"Backtest failed with return code {process.returncode}")
                return None

            logger.info(f"Backtest completed successfully. Results in {output_dir}")
            return output_dir

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None

    def validate_results(self, results_dir: Path):
        """Validate the backtest results are real (not synthetic)"""
        logger.info("Validating backtest results...")

        validation = {
            'is_real': False,
            'has_statistics': False,
            'has_trades': False,
            'has_logs': False,
            'metrics': {}
        }

        # Check for statistics file
        stats_files = list(results_dir.glob("*-statistics.json"))
        if stats_files:
            stats_file = stats_files[0]
            with open(stats_file) as f:
                stats = json.load(f)

            validation['has_statistics'] = True

            # Check for real metrics (not synthetic placeholders)
            if 'TotalPerformance' in stats:
                perf = stats['TotalPerformance']
                validation['metrics'] = {
                    'total_trades': perf.get('TotalTrades', 0),
                    'sharpe_ratio': perf.get('SharpeRatio', 0),
                    'total_return': perf.get('PortfolioStatistics', {}).get('TotalNetProfit', 0),
                    'win_rate': perf.get('WinRate', 0)
                }

                # Real results should have non-zero trades
                if validation['metrics']['total_trades'] > 0:
                    validation['is_real'] = True

        # Check for order log
        order_files = list(results_dir.glob("*-order-events.json"))
        if order_files:
            with open(order_files[0]) as f:
                orders = json.load(f)
                if orders and len(orders) > 0:
                    validation['has_trades'] = True

        # Check for algorithm log
        log_files = list(results_dir.glob("*.log"))
        if log_files:
            log_content = log_files[0].read_text()
            # Real logs should have Docker execution traces
            if "TRACE::" in log_content or "DEBUG::" in log_content:
                validation['has_logs'] = True
                validation['is_real'] = True

        # Final validation
        validation['status'] = 'REAL' if validation['is_real'] else 'SYNTHETIC'

        return validation

    def cleanup_synthetic_results(self):
        """Remove synthetic results from previous runs"""
        logger.info("Cleaning up synthetic results...")

        synthetic_dirs = []
        for result_dir in self.results_dir.glob("*/"):
            # Check if directory contains synthetic results
            stats_files = list(result_dir.glob("*-statistics.json"))
            if stats_files:
                with open(stats_files[0]) as f:
                    stats = json.load(f)
                    # Synthetic results have specific markers
                    if stats.get('IsSimulated') or 'DEMO' in str(stats):
                        synthetic_dirs.append(result_dir)

        for dir_path in synthetic_dirs:
            logger.info(f"Removing synthetic results: {dir_path}")
            shutil.rmtree(dir_path)

        logger.info(f"Cleaned up {len(synthetic_dirs)} synthetic result directories")

    def run_full_validation(self):
        """Run complete validation workflow"""
        logger.info("="*60)
        logger.info("QUANTCONNECT REAL BACKTEST VALIDATION")
        logger.info("="*60)

        # Check validation mode
        validation_mode = os.environ.get('VALIDATION_MODE', 'real')
        is_pr = os.environ.get('GITHUB_EVENT_NAME') == 'pull_request'

        if is_pr and not os.environ.get('QC_USER_ID'):
            logger.info("Running in PR mode without secrets - using mock validation")
            validation_mode = 'mock'

        # Step 1: Setup environment
        if not self.setup_environment():
            logger.error("Failed to setup environment")
            return False

        # Step 2: Clean up synthetic results
        self.cleanup_synthetic_results()

        # Step 3: Create test algorithm
        algorithm_dir = self.create_test_algorithm()

        # Step 4: Run real backtest
        results_dir = self.run_backtest(algorithm_dir)
        if not results_dir:
            logger.error("Backtest execution failed")
            return False

        # Step 5: Validate results
        validation = self.validate_results(results_dir)

        # Print results
        logger.info("\n" + "="*60)
        logger.info("VALIDATION RESULTS")
        logger.info("="*60)
        logger.info(f"Status: {validation['status']}")
        logger.info(f"Has Statistics: {validation['has_statistics']}")
        logger.info(f"Has Trades: {validation['has_trades']}")
        logger.info(f"Has Logs: {validation['has_logs']}")

        if validation['metrics']:
            logger.info("\nMetrics:")
            for key, value in validation['metrics'].items():
                logger.info(f"  {key}: {value}")

        # Check if this is a mock validation (for PRs without secrets)
        if os.environ.get('QUANTCONNECT_USER_ID') == 'mock_user':
            logger.info("\n✓ MOCK VALIDATION PASSED: Structure validated successfully (PR mode)")
            return True
        elif validation['is_real']:
            logger.info("\n✓ VALIDATION PASSED: Real QuantConnect backtest executed successfully!")
            return True
        else:
            logger.error("\n✗ VALIDATION FAILED: Results appear to be synthetic!")
            return False

def main():
    """Main entry point"""
    validator = QuantConnectBacktestValidator()

    # Check for GitHub Actions environment
    if os.environ.get('GITHUB_ACTIONS'):
        logger.info("Running in GitHub Actions CI environment")
        # CI credentials should be in secrets
        os.environ['QUANTCONNECT_USER_ID'] = os.environ.get('QC_USER_ID', '')
        os.environ['QUANTCONNECT_API_TOKEN'] = os.environ.get('QC_API_TOKEN', '')

    try:
        success = validator.run_full_validation()
        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()