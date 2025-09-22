
from typing import Dict, Any, List
import numpy as np
from datetime import datetime, timezone

class MathematicalIntegration:
    """Integrates mathematical toolkits into trading workflow"""

    def __init__(self, hmm_server, ou_server, signal_server):
        self.hmm_server = hmm_server
        self.ou_server = ou_server
        self.signal_server = signal_server

    async def analyze_market_regime(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete market analysis using all mathematical tools"""

        # Step 1: Detect market regime using HMM
        regime_result = await self.hmm_server.detect_regime({
            "returns": market_data["returns"],
            "lookback_days": 60
        })

        current_regime = regime_result["current_regime"]

        # Step 2: Apply OU mean reversion analysis based on regime
        ou_params = {
            "prices": market_data["prices"],
            "regime": current_regime,
            "confidence_level": 0.95
        }
        mean_reversion = await self.ou_server.analyze_mean_reversion(ou_params)

        # Step 3: Apply signal processing for noise reduction
        # Use wavelet for trending regimes, Fourier for ranging
        if current_regime in ["trending", "bull", "bear"]:
            signal_params = {
                "signal": market_data["raw_signals"],
                "method": "wavelet",
                "level": 3
            }
        else:
            signal_params = {
                "signal": market_data["raw_signals"],
                "method": "fourier",
                "cutoff_frequency": 0.1
            }

        filtered_signals = await self.signal_server.filter_signals(signal_params)

        # Step 4: Combine all analyses
        return {
            "regime": {
                "current": current_regime,
                "confidence": regime_result["confidence"],
                "transition_probability": regime_result.get("transition_prob", 0)
            },
            "mean_reversion": {
                "half_life": mean_reversion["half_life"],
                "reversion_speed": mean_reversion["reversion_speed"],
                "target_price": mean_reversion["target_price"],
                "z_score": mean_reversion["z_score"]
            },
            "signals": {
                "filtered": filtered_signals["filtered_signal"],
                "signal_to_noise": filtered_signals["snr"],
                "confidence": filtered_signals["confidence"]
            },
            "recommendation": self._generate_recommendation(
                current_regime, mean_reversion, filtered_signals
            )
        }

    def _generate_recommendation(self, regime, mean_reversion, signals):
        """Generate trading recommendation based on all analyses"""

        # Strong mean reversion signal in ranging market
        if regime == "ranging" and abs(mean_reversion["z_score"]) > 2:
            if mean_reversion["z_score"] > 2:
                return "SELL"  # Overbought
            else:
                return "BUY"   # Oversold

        # Trend following in trending market
        elif regime in ["bull", "trending_up"] and signals["confidence"] > 0.7:
            return "BUY"
        elif regime in ["bear", "trending_down"] and signals["confidence"] > 0.7:
            return "SELL"

        # Low confidence or transitioning regime
        else:
            return "HOLD"
