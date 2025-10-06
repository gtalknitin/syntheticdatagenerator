#!/usr/bin/env python3
"""
V10 Real-Enhanced Synthetic Data Generator

CRITICAL FIXES FROM V9:
1. Uses REAL NIFTY spot prices (not simulated)
2. Uses REAL India VIX data (not regime-based)
3. FIXED Greeks calculations (correct deltas for deep ITM/OTM)
4. FIXED volume decay (exponential, not Gaussian)
5. FIXED bid-ask spreads (distance + liquidity based)
6. Comprehensive validation built-in

Author: NikAlgoBulls
Date: October 5, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Tuple, List, Dict
from scipy.stats import norm

from synthetic_data_generator.io.seed_data_loader import NiftySeedDataLoader

logger = logging.getLogger(__name__)


class V10RealEnhancedGenerator:
    """
    V10 Generator using REAL market data as foundation

    Key Improvements:
    - Real underlying prices (not simulated random walk)
    - Real VIX data (not regime-based simulation)
    - Mathematically correct Greeks
    - Realistic liquidity modeling
    """

    def __init__(self):
        """Initialize V10 generator with real seed data"""

        # Output directory
        self.base_path = Path('data/generated/v10_real_enhanced/hourly')
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Load REAL seed data
        print("📥 Loading real market data...")
        self.seed_loader = NiftySeedDataLoader()
        self.nifty_1min = self.seed_loader.load()
        self.nifty_hourly = self._aggregate_to_hourly(self.nifty_1min)

        # Load REAL VIX data
        vix_path = Path('data/seed/india_vix.csv')
        self.vix_data = pd.read_csv(vix_path)
        self.vix_data['date'] = pd.to_datetime(self.vix_data['date']).dt.date

        print(f"  ✓ NIFTY: {len(self.nifty_hourly)} hourly candles")
        print(f"  ✓ VIX: {len(self.vix_data)} days")

        # Market parameters (UNCHANGED - standard values)
        self.risk_free_rate = 0.065  # 6.5% RBI repo rate
        self.strike_interval = 50

        # Strike range (wider to accommodate price movements)
        self.min_strike = 18000
        self.max_strike = 30000

        # Expiry schedule (same as V9)
        self.expiry_schedule = self._load_expiry_schedule()

        # Stats tracking
        self.stats = {
            'files_created': 0,
            'total_rows': 0,
            'dates_processed': [],
            'greeks_validation': {
                'deep_itm_passed': 0,
                'deep_otm_passed': 0,
                'atm_passed': 0,
                'total_checks': 0
            },
            'volume_validation': {
                'far_strikes_low_volume': 0,
                'atm_high_volume': 0,
                'total_checks': 0
            },
            'quality_issues': []
        }

    def _aggregate_to_hourly(self, df_1min: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1-minute NIFTY data to hourly candles

        FIX: Uses REAL price movements (not simulated)

        Hour bins:
        - H1: 09:15-10:15
        - H2: 10:15-11:15
        - H3: 11:15-12:15
        - H4: 12:15-13:15
        - H5: 13:15-14:15
        - H6: 14:15-15:15
        - H7: 15:15-15:30 (last 15 mins)
        """
        df = df_1min.copy()

        # Create hour bins
        df['hour_bin'] = df['date'].dt.floor('H')

        # Aggregate OHLCV
        hourly = df.groupby('hour_bin').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()

        hourly.rename(columns={'hour_bin': 'timestamp'}, inplace=True)

        # Add trading date
        hourly['trading_date'] = hourly['timestamp'].dt.date

        logger.info(f"Aggregated {len(df_1min)} 1-min candles to {len(hourly)} hourly candles")

        return hourly

    def _load_expiry_schedule(self) -> dict:
        """
        Load NIFTY expiry schedule

        Using realistic 2024-2025 calendar (covers full seed data range)
        """
        return {
            'weekly': {
                1: [4, 11, 18, 25],           # Jan 2024
                2: [1, 8, 15, 22, 29],        # Feb
                3: [7, 14, 21, 28],           # Mar
                4: [4, 11, 18, 25],           # Apr
                5: [2, 9, 16, 23, 30],        # May
                6: [6, 13, 20, 27],           # Jun
                7: [4, 11, 18, 25],           # Jul
                8: [1, 8, 15, 22, 29],        # Aug
                9: [5, 12, 19, 26],           # Sep
                10: [3, 10, 17, 24, 31],      # Oct
                11: [7, 14, 21, 28],          # Nov 2024
                12: [5, 12, 19, 26],          # Dec 2024
                # 2025
                13: [2, 9, 16, 23, 30],       # Jan 2025 (month 1 of 2025)
                14: [6, 13, 20, 27],          # Feb 2025
                15: [6, 13, 20, 27],          # Mar 2025
                16: [3, 10, 17, 24],          # Apr 2025
                17: [1, 8, 15, 22, 29],       # May 2025
                18: [5, 12, 19, 26],          # Jun 2025
                19: [3, 10, 17, 24, 31],      # Jul 2025
                20: [7, 14, 21, 28],          # Aug 2025
                21: [4, 11, 18, 25],          # Sep 2025
                22: [2, 9, 16, 23, 30]        # Oct 2025
            },
            'monthly': {
                1: 25,   # Jan 2024
                2: 29,   # Feb
                3: 28,   # Mar
                4: 25,   # Apr
                5: 30,   # May
                6: 27,   # Jun
                7: 25,   # Jul
                8: 29,   # Aug
                9: 26,   # Sep
                10: 31,  # Oct
                11: 28,  # Nov 2024
                12: 26,  # Dec 2024
                # 2025
                13: 30,  # Jan 2025 (use as month 13 for 2025)
                14: 27,  # Feb 2025
                15: 27,  # Mar 2025
                16: 24,  # Apr 2025
                17: 29,  # May 2025
                18: 26,  # Jun 2025
                19: 31,  # Jul 2025
                20: 28,  # Aug 2025
                21: 25,  # Sep 2025
                22: 30   # Oct 2025
            }
        }

    def get_vix_for_date(self, date: datetime.date) -> float:
        """
        Get REAL VIX value for a date

        FIX: Uses actual India VIX data (not simulated regime)
        """
        # Find matching date in VIX data
        vix_row = self.vix_data[self.vix_data['date'] == date]

        if not vix_row.empty:
            return float(vix_row.iloc[0]['vix_close'])

        # Fallback: use nearest date
        self.vix_data['date_diff'] = abs((pd.to_datetime(self.vix_data['date']) - pd.to_datetime(date)).dt.days)
        nearest = self.vix_data.nsmallest(1, 'date_diff')

        if not nearest.empty:
            vix = float(nearest.iloc[0]['vix_close'])
            logger.warning(f"VIX for {date} not found, using nearest: {vix:.2f}")
            return vix

        # Last resort: use mean VIX
        return float(self.vix_data['vix_close'].mean())

    def get_active_expiries(self, date: datetime.date) -> List[Tuple[datetime.date, str]]:
        """
        Get active expiries for a date

        Returns list of (expiry_date, expiry_type) tuples
        """
        expiries = []

        # Weekly expiries (next 4 weeks)
        current = datetime.combine(date, datetime.min.time())
        for _ in range(4):
            # Find next Thursday
            days_ahead = (3 - current.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            expiry_date = current + timedelta(days=days_ahead)

            if expiry_date.date() >= date:
                expiries.append((expiry_date.date(), 'weekly'))

            current = expiry_date + timedelta(days=1)

        # Monthly expiries (current + next 2 months)
        for month_offset in range(3):
            target_month = date.month + month_offset
            target_year = date.year

            while target_month > 12:
                target_month -= 12
                target_year += 1

            # Map to schedule key (2025 months start at 13)
            if target_year == 2024:
                schedule_key = target_month
            else:  # 2025
                schedule_key = target_month + 12

            if schedule_key in self.expiry_schedule['monthly']:
                day = self.expiry_schedule['monthly'][schedule_key]
                try:
                    expiry_date = datetime(target_year, target_month, day).date()
                    if expiry_date >= date:
                        expiries.append((expiry_date, 'monthly'))
                except ValueError:
                    # Invalid date (e.g., Feb 30), skip
                    logger.warning(f"Invalid expiry date: {target_year}-{target_month}-{day}")
                    continue

        # CRITICAL FIX: Remove duplicates where same date has both weekly and monthly
        # If a date appears as both weekly and monthly, keep only monthly
        unique_expiries = {}
        for expiry_date, expiry_type in expiries:
            if expiry_date in unique_expiries:
                # Keep monthly over weekly
                if expiry_type == 'monthly':
                    unique_expiries[expiry_date] = expiry_type
            else:
                unique_expiries[expiry_date] = expiry_type

        # Convert back to list of tuples and sort
        expiries = sorted([(date, etype) for date, etype in unique_expiries.items()])

        # Limit to 6 expiries
        return expiries[:6]

    def calculate_iv(self, spot: float, strike: int, tte_days: int,
                    expiry_type: str, vix: float) -> float:
        """
        Calculate implied volatility with REALISTIC smile

        CRITICAL FIX FROM V9:
        - Symmetric smile (both wings have higher IV)
        - Quadratic shape (not linear)
        - Proper term structure

        IV Smile Theory:
        - ATM options have LOWEST IV (most liquid, fair pricing)
        - OTM options have HIGHER IV (tail risk premium)
        - Deep OTM have VERY HIGH IV (lottery tickets)
        """

        # Base IV from REAL VIX
        atm_iv = vix / 100.0

        # Calculate log-moneyness (standard for smile)
        # log(S/K) = 0 at ATM, negative for OTM calls, positive for ITM calls
        log_moneyness = np.log(spot / strike)

        # QUADRATIC smile (symmetric U-shape)
        # This makes ATM cheapest, wings more expensive
        smile_adjustment = 0.15 * log_moneyness**2

        # Additional adjustment for very far strikes
        distance_pct = abs(strike - spot) / spot
        if distance_pct > 0.15:  # >15% from ATM
            # Steep increase for far OTM/ITM (tail risk)
            smile_adjustment += 0.3 * (distance_pct - 0.15)

        iv = atm_iv * (1 + smile_adjustment)

        # Term structure (shorter TTE = higher IV)
        if tte_days <= 2:
            iv *= 1.25  # +25% for weekend/expiry gamma
        elif tte_days <= 7:
            iv *= 1.15  # +15% for weekly options
        elif tte_days <= 15:
            iv *= 1.08  # +8% for mid-term
        elif tte_days <= 30:
            iv *= 1.03  # +3% for monthly
        # else: base IV for longer dated

        # Expiry-type specific (weekly options more volatile near expiry)
        if expiry_type == 'weekly' and tte_days <= 3:
            iv *= 1.10  # Additional 10% vol for expiry week

        # Small random noise (±2% for realism)
        iv *= (1 + np.random.uniform(-0.02, 0.02))

        # Realistic bounds (India VIX typically 8%-60%)
        return np.clip(iv, 0.08, 0.60)

    def black_scholes(self, S: float, K: float, T: float, sigma: float,
                     option_type: str) -> dict:
        """
        Black-Scholes option pricing with CORRECT Greeks

        CRITICAL FIX FROM V9:
        - Deep ITM options now have delta ~0.95-0.99 (not 0.67!)
        - Deep OTM options now have delta ~0.01-0.05 (not inflated)
        - ATM options have delta ~0.50 (correct)

        The issue in V9 was incorrect IV calculation, not BS formula.
        This implementation is mathematically correct.

        Returns: {price, delta, gamma, theta, vega}
        """

        # Handle expiry (T=0)
        if T <= 0:
            if option_type == 'CE':
                price = max(S - K, 0)
                delta = 1.0 if S > K else 0.0
            else:  # PE
                price = max(K - S, 0)
                delta = -1.0 if K > S else 0.0

            return {
                'price': round(max(price, 0.05), 2),
                'delta': delta,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }

        # Standard Black-Scholes calculation
        sqrt_T = np.sqrt(T)

        # d1 and d2
        d1 = (np.log(S/K) + (self.risk_free_rate + 0.5*sigma**2)*T) / (sigma*sqrt_T)
        d2 = d1 - sigma*sqrt_T

        # Price
        if option_type == 'CE':
            price = S * norm.cdf(d1) - K * np.exp(-self.risk_free_rate*T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:  # PE
            price = K * np.exp(-self.risk_free_rate*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)

        # Greeks
        gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
        vega = S * norm.pdf(d1) * sqrt_T / 100  # Per 1% IV change

        # Theta (per day, always negative for long options)
        theta_annual = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
        if option_type == 'CE':
            theta_annual -= self.risk_free_rate * K * np.exp(-self.risk_free_rate*T) * norm.cdf(d2)
        else:
            theta_annual += self.risk_free_rate * K * np.exp(-self.risk_free_rate*T) * norm.cdf(-d2)
        theta = theta_annual / 365  # Per day

        return {
            'price': round(max(price, 0.05), 2),  # Min ₹0.05
            'delta': round(delta, 4),
            'gamma': round(max(gamma, 0), 6),
            'theta': round(theta, 2),
            'vega': round(vega, 2)
        }

    def generate_volume(self, spot: float, strike: int, tte_days: int,
                       expiry_type: str, hour_index: int, vix: float) -> int:
        """
        Generate REALISTIC volume with steep decay

        CRITICAL FIX FROM V9:
        - Far strikes now have <2% of ATM volume (not 94%!)
        - Uses exponential decay based on DISTANCE in points
        - Expiry week volume surge
        - Intraday U-shaped pattern

        Example decay:
        - ATM (0 pts):    100% volume
        - 250 pts away:    37% volume
        - 500 pts away:    13% volume
        - 1000 pts away:    2% volume
        - 2000 pts away:  0.03% volume (almost nothing)
        """

        # Base volume by expiry type
        if expiry_type == 'monthly':
            base_volume = 8000  # Contracts per hour
        else:
            base_volume = 3000

        # Distance-based decay (CRITICAL FIX!)
        distance = abs(strike - spot)

        if distance < 100:  # Very near ATM
            distance_factor = 1.0
        elif distance < 500:  # Near ATM
            distance_factor = np.exp(-0.004 * distance)
        elif distance < 1500:  # Medium distance
            distance_factor = np.exp(-0.008 * distance)
        else:  # Far OTM/ITM
            distance_factor = np.exp(-0.012 * distance)

        # Time to expiry factor (volume surge near expiry)
        if tte_days <= 1:
            tte_factor = 4.0  # 4x on expiry day
        elif tte_days <= 3:
            tte_factor = 2.5  # 2.5x in last 3 days
        elif tte_days <= 7:
            tte_factor = 1.5  # 1.5x in expiry week
        else:
            tte_factor = 1.0

        # Intraday pattern (U-shaped: high at open/close)
        intraday_factors = [1.8, 1.2, 0.9, 0.7, 0.9, 1.3, 1.5]
        intraday_factor = intraday_factors[min(hour_index, 6)]

        # VIX boost (higher volatility = higher volume)
        vix_factor = 1.0 + (max(vix - 15, 0) / 50)

        # Random variation (log-normal for realism)
        random_factor = np.random.lognormal(0, 0.3)

        # Calculate final volume
        volume = int(
            base_volume *
            distance_factor *
            tte_factor *
            intraday_factor *
            vix_factor *
            random_factor
        )

        return max(volume, 5)  # Minimum 5 contracts

    def generate_oi(self, spot: float, strike: int, tte_days: int,
                   expiry_type: str) -> int:
        """
        Generate realistic Open Interest

        OI characteristics:
        - Accumulates over time (higher than volume)
        - Concentrates at ATM
        - Decays as expiry approaches (positions closed)
        """

        # Base OI (higher than volume as it accumulates)
        if expiry_type == 'monthly':
            base_oi = 150000
        else:
            base_oi = 60000

        # Distance decay (same concept as volume, but less steep)
        distance = abs(strike - spot)
        if distance < 100:
            distance_factor = 1.0
        elif distance < 500:
            distance_factor = np.exp(-0.003 * distance)
        elif distance < 1500:
            distance_factor = np.exp(-0.006 * distance)
        else:
            distance_factor = np.exp(-0.009 * distance)

        # OI decays as expiry approaches (positions closed)
        if tte_days <= 1:
            tte_factor = 0.3  # 70% closed on expiry
        elif tte_days <= 3:
            tte_factor = 0.6  # 40% closed in last 3 days
        elif tte_days <= 7:
            tte_factor = 0.9
        else:
            # OI builds up over time
            tte_factor = 1.0 + (tte_days / 30) * 0.2

        # Random variation
        random_factor = np.random.lognormal(0, 0.25)

        oi = int(base_oi * distance_factor * tte_factor * random_factor)

        return max(oi, 10)

    def generate_bid_ask(self, price: float, spot: float, strike: int,
                        volume: int, oi: int) -> Tuple[float, float]:
        """
        Generate REALISTIC bid-ask spreads

        CRITICAL FIX FROM V9:
        - Wider spreads for far strikes
        - Wider spreads for low liquidity
        - Minimum tick size ₹0.05

        Spread determinants:
        1. Price level (cheaper options = wider % spread)
        2. Distance from ATM (far strikes = wider spread)
        3. Liquidity (low volume = wider spread)
        """

        # Base spread as percentage of price
        if price < 5:
            base_spread_pct = 0.15  # 15% for very cheap options
        elif price < 20:
            base_spread_pct = 0.08  # 8%
        elif price < 50:
            base_spread_pct = 0.04  # 4%
        elif price < 200:
            base_spread_pct = 0.02  # 2%
        else:
            base_spread_pct = 0.01  # 1% for expensive options

        # Distance adjustment (CRITICAL: wider for far strikes)
        distance_pct = abs(strike - spot) / spot
        if distance_pct > 0.10:  # >10% from ATM
            base_spread_pct *= 2.5  # Much wider
        elif distance_pct > 0.05:  # >5% from ATM
            base_spread_pct *= 1.5

        # Liquidity adjustment (low volume = wide spreads)
        if volume < 50:
            base_spread_pct *= 3.0  # Very illiquid
        elif volume < 200:
            base_spread_pct *= 2.0
        elif volume < 1000:
            base_spread_pct *= 1.3

        # Calculate spread
        spread = max(price * base_spread_pct, 0.05)  # Min ₹0.05

        # Bid/Ask (ensure bid < price < ask)
        mid = price
        bid = round(mid - spread/2, 1)  # Round to ₹0.10
        ask = round(mid + spread/2, 1)

        # Ensure minimum tick and valid relationship
        bid = max(bid, 0.05)
        ask = max(ask, bid + 0.05)

        return (bid, ask)

    def generate_ohlc(self, spot_ohlc: dict, strike: int, option_type: str,
                     T: float, iv: float) -> dict:
        """
        Generate option OHLC from underlying OHLC

        Methodology:
        - Calculate option price at each point of underlying OHLC
        - For calls: high underlying -> high option price
        - For puts: high underlying -> low option price (inverse)
        """

        # Calculate option price at each underlying level
        prices = []
        for spot in [spot_ohlc['open'], spot_ohlc['high'],
                    spot_ohlc['low'], spot_ohlc['close']]:

            option_data = self.black_scholes(
                S=spot,
                K=strike,
                T=T,
                sigma=iv,
                option_type=option_type
            )
            prices.append(option_data['price'])

        # Map to OHLC based on option type
        if option_type == 'CE':
            # Calls: direct relationship with underlying
            return {
                'open': prices[0],
                'high': max(prices),  # When underlying is high
                'low': min(prices),   # When underlying is low
                'close': prices[3]
            }
        else:  # PE
            # Puts: inverse relationship
            return {
                'open': prices[0],
                'high': max(prices),  # When underlying is low
                'low': min(prices),   # When underlying is high
                'close': prices[3]
            }

    def validate_greeks(self, greeks: dict, spot: float, strike: int,
                       option_type: str):
        """
        Validate Greeks are REALISTIC

        CRITICAL CHECKS (fixing V9 issues):
        - Deep ITM (>2000 pts): Delta > 0.90
        - Deep OTM (>2000 pts): Delta < 0.10
        - ATM (±100 pts): Delta ~0.50
        """

        delta = greeks['delta']
        distance = strike - spot if option_type == 'CE' else spot - strike

        # Check 1: Deep ITM should have high delta
        if option_type == 'CE' and strike < spot - 2000:  # Deep ITM call
            if delta > 0.90:
                self.stats['greeks_validation']['deep_itm_passed'] += 1
            else:
                issue = f"Deep ITM CE delta too low: K={strike}, S={spot:.0f}, delta={delta:.2f}"
                logger.warning(issue)
                self.stats['quality_issues'].append(issue)

        elif option_type == 'PE' and strike > spot + 2000:  # Deep ITM put
            if delta < -0.90:
                self.stats['greeks_validation']['deep_itm_passed'] += 1
            else:
                issue = f"Deep ITM PE delta too high: K={strike}, S={spot:.0f}, delta={delta:.2f}"
                logger.warning(issue)
                self.stats['quality_issues'].append(issue)

        # Check 2: Deep OTM should have low delta
        elif option_type == 'CE' and strike > spot + 2000:  # Deep OTM call
            if abs(delta) < 0.10:
                self.stats['greeks_validation']['deep_otm_passed'] += 1
            else:
                issue = f"Deep OTM CE delta too high: K={strike}, S={spot:.0f}, delta={delta:.2f}"
                logger.warning(issue)
                self.stats['quality_issues'].append(issue)

        elif option_type == 'PE' and strike < spot - 2000:  # Deep OTM put
            if abs(delta) < 0.10:
                self.stats['greeks_validation']['deep_otm_passed'] += 1
            else:
                issue = f"Deep OTM PE delta too high: K={strike}, S={spot:.0f}, delta={delta:.2f}"
                logger.warning(issue)
                self.stats['quality_issues'].append(issue)

        # Check 3: ATM should have delta ≈ 0.5
        elif abs(strike - spot) < 100:  # ATM
            if 0.40 < abs(delta) < 0.60:
                self.stats['greeks_validation']['atm_passed'] += 1
            else:
                issue = f"ATM delta off: K={strike}, S={spot:.0f}, delta={delta:.2f}"
                logger.warning(issue)
                self.stats['quality_issues'].append(issue)

        self.stats['greeks_validation']['total_checks'] += 1

        # Bounds checks
        assert -1.0 <= delta <= 1.0, f"Delta out of bounds: {delta}"
        assert greeks['gamma'] >= 0, f"Gamma negative: {greeks['gamma']}"
        assert greeks['vega'] >= 0, f"Vega negative: {greeks['vega']}"

    def get_strikes_for_spot(self, spot: float) -> List[int]:
        """
        Get filtered strikes for a spot price

        Strike filtering (to reduce data size):
        - All strikes within ±500 points (ATM region)
        - Every 2nd strike (₹100 interval) for ±500 to ±1500
        - Every 4th strike (₹200 interval) beyond ±1500
        """
        all_strikes = list(range(self.min_strike, self.max_strike + 1, self.strike_interval))

        filtered = []
        for strike in all_strikes:
            distance = abs(strike - spot)

            if distance <= 500:
                # ATM region: all strikes
                filtered.append(strike)
            elif distance <= 1500:
                # Near region: every 2nd strike (₹100 interval)
                if (strike - self.min_strike) % 100 == 0:
                    filtered.append(strike)
            else:
                # Far region: every 4th strike (₹200 interval)
                if (strike - self.min_strike) % 200 == 0:
                    filtered.append(strike)

        return filtered

    def generate_day_data(self, date: datetime.date) -> pd.DataFrame:
        """
        Generate complete options data for ONE day

        Process:
        1. Get real NIFTY hourly prices for the day
        2. Get real VIX for the day
        3. Determine active expiries
        4. For each hour:
             For each expiry:
                 For each strike:
                     For each option type (CE/PE):
                         - Calculate IV (with smile)
                         - Calculate Greeks (Black-Scholes)
                         - Generate OHLC (from underlying)
                         - Generate volume/OI (with decay)
                         - Generate bid/ask (with spreads)
                         - Validate
        """

        # Get real data for this date
        day_hourly = self.nifty_hourly[self.nifty_hourly['trading_date'] == date]

        if day_hourly.empty:
            logger.warning(f"No NIFTY data for {date}")
            return pd.DataFrame()

        vix = self.get_vix_for_date(date)
        expiries = self.get_active_expiries(date)

        rows = []

        # For each hourly candle (7 per day)
        for hour_idx, (idx, hour_row) in enumerate(day_hourly.iterrows()):
            timestamp = hour_row['timestamp']

            spot_ohlc = {
                'open': hour_row['open'],
                'high': hour_row['high'],
                'low': hour_row['low'],
                'close': hour_row['close']
            }
            spot_close = spot_ohlc['close']

            # Get strikes for this spot level
            strikes = self.get_strikes_for_spot(spot_close)

            # For each expiry
            for expiry_date, expiry_type in expiries:
                tte_days = (expiry_date - date).days
                tte_years = tte_days / 365.0

                # For each strike
                for strike in strikes:
                    # For each option type
                    for option_type in ['CE', 'PE']:

                        # Calculate IV with realistic smile
                        iv = self.calculate_iv(
                            spot=spot_close,
                            strike=strike,
                            tte_days=tte_days,
                            expiry_type=expiry_type,
                            vix=vix
                        )

                        # Calculate Greeks using Black-Scholes
                        greeks = self.black_scholes(
                            S=spot_close,
                            K=strike,
                            T=tte_years,
                            sigma=iv,
                            option_type=option_type
                        )

                        # Generate OHLC from underlying movement
                        ohlc = self.generate_ohlc(
                            spot_ohlc=spot_ohlc,
                            strike=strike,
                            option_type=option_type,
                            T=tte_years,
                            iv=iv
                        )

                        # Generate volume
                        volume = self.generate_volume(
                            spot=spot_close,
                            strike=strike,
                            tte_days=tte_days,
                            expiry_type=expiry_type,
                            hour_index=hour_idx,
                            vix=vix
                        )

                        # Generate OI
                        oi = self.generate_oi(
                            spot=spot_close,
                            strike=strike,
                            tte_days=tte_days,
                            expiry_type=expiry_type
                        )

                        # Generate bid/ask
                        bid, ask = self.generate_bid_ask(
                            price=ohlc['close'],
                            spot=spot_close,
                            strike=strike,
                            volume=volume,
                            oi=oi
                        )

                        # Validate Greeks (only for first hour to avoid spam)
                        if hour_idx == 0:
                            self.validate_greeks(greeks, spot_close, strike, option_type)

                        # Create row
                        row = {
                            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            'symbol': 'NIFTY',
                            'strike': strike,
                            'option_type': option_type,
                            'expiry': expiry_date.strftime('%Y-%m-%d'),
                            'expiry_type': expiry_type,
                            'open': ohlc['open'],
                            'high': ohlc['high'],
                            'low': ohlc['low'],
                            'close': ohlc['close'],
                            'volume': volume,
                            'oi': oi,
                            'bid': bid,
                            'ask': ask,
                            'iv': round(iv, 4),
                            'delta': greeks['delta'],
                            'gamma': greeks['gamma'],
                            'theta': greeks['theta'],
                            'vega': greeks['vega'],
                            'underlying_price': round(spot_close, 2),
                            'vix': round(vix, 2)
                        }

                        rows.append(row)

        return pd.DataFrame(rows)

    def validate_day_data(self, df: pd.DataFrame) -> dict:
        """
        Validate generated data for completeness and quality
        """
        if df.empty:
            return {'valid': False, 'issues': ['Empty dataframe']}

        issues = []

        # Check for missing values
        if df.isnull().any().any():
            issues.append(f"Missing values: {df.isnull().sum().to_dict()}")

        # Check OHLC validity
        invalid = df[
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ]
        if len(invalid) > 0:
            issues.append(f"Invalid OHLC in {len(invalid)} rows")

        # Check bid/ask validity
        invalid_spread = df[(df['bid'] >= df['ask']) | (df['bid'] > df['close'])]
        if len(invalid_spread) > 0:
            issues.append(f"Invalid bid/ask in {len(invalid_spread)} rows")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'pass_rate': 1.0 - (len(invalid) + len(invalid_spread)) / len(df)
        }

    def generate_all_data(self):
        """
        Main generation loop for ALL 438 trading days
        """
        print("="*80)
        print("V10 REAL-ENHANCED SYNTHETIC DATA GENERATOR")
        print("="*80)
        print(f"Using REAL NIFTY prices + REAL VIX data")
        print(f"Output: {self.base_path}")
        print("="*80)

        # Get all trading dates from real NIFTY data
        trading_dates = sorted(self.nifty_hourly['trading_date'].unique())

        print(f"\n📊 Generating options data for {len(trading_dates)} days...")

        for i, date in enumerate(trading_dates, 1):
            print(f"  [{i:3}/{len(trading_dates)}] {date}", end=' ... ')

            # Generate data
            df = self.generate_day_data(date)

            if df.empty:
                print("⊘ No data")
                continue

            # Validate
            validation = self.validate_day_data(df)

            # Save
            filename = f"NIFTY_OPTIONS_1H_{date.strftime('%Y%m%d')}.csv"
            filepath = self.base_path / filename
            df.to_csv(filepath, index=False)

            # Update stats
            self.stats['files_created'] += 1
            self.stats['total_rows'] += len(df)
            self.stats['dates_processed'].append(str(date))

            # Print status
            vix = df['vix'].iloc[0]
            strikes = df['strike'].nunique()
            status = "✓" if validation['valid'] else "⚠"
            print(f"{status} ({len(df):,} rows, {strikes} strikes, VIX: {vix:.1f})")

        # Save metadata
        self.save_metadata()

        # Print summary
        self.print_summary()

        return self.stats

    def save_metadata(self):
        """Save generation metadata"""
        metadata_path = self.base_path.parent / 'metadata'
        metadata_path.mkdir(exist_ok=True)

        # Calculate validation pass rates
        gv = self.stats['greeks_validation']
        pass_rates = {}
        if gv['total_checks'] > 0:
            pass_rates = {
                'deep_itm': gv['deep_itm_passed'] / max(gv['total_checks'], 1),
                'deep_otm': gv['deep_otm_passed'] / max(gv['total_checks'], 1),
                'atm': gv['atm_passed'] / max(gv['total_checks'], 1)
            }

        metadata = {
            'version': '10.0-RealEnhanced',
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_sources': {
                'underlying': 'Real NIFTY 1-minute data (Jan 2024 - Oct 2025)',
                'vix': 'Real India VIX from Yahoo Finance',
                'seed_files': {
                    'nifty': 'data/seed/nifty_data_min.csv',
                    'vix': 'data/seed/india_vix.csv'
                }
            },
            'critical_fixes_from_v9': [
                'Uses REAL underlying prices (not simulated)',
                'Uses REAL VIX data (not regime-based)',
                'FIXED Greeks: Deep ITM delta > 0.90',
                'FIXED Greeks: Deep OTM delta < 0.10',
                'FIXED Volume: Exponential decay (not Gaussian)',
                'FIXED Spreads: Distance + liquidity based'
            ],
            'period': {
                'start': self.stats['dates_processed'][0] if self.stats['dates_processed'] else None,
                'end': self.stats['dates_processed'][-1] if self.stats['dates_processed'] else None,
                'trading_days': len(self.stats['dates_processed'])
            },
            'specifications': {
                'frequency': '1HR',
                'timeframe': '1H',  # Explicit timeframe for clarity
                'candles_per_day': 7,
                'strike_range': [self.min_strike, self.max_strike],
                'strike_interval': self.strike_interval
            },
            'stats': self.stats,
            'validation': {
                'greeks_pass_rates': pass_rates,
                'total_quality_issues': len(self.stats['quality_issues'])
            }
        }

        with open(metadata_path / 'generation_info.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save quality issues if any
        if self.stats['quality_issues']:
            with open(metadata_path / 'quality_issues.txt', 'w') as f:
                for issue in self.stats['quality_issues'][:100]:  # First 100
                    f.write(issue + '\n')

    def print_summary(self):
        """Print generation summary"""
        print("\n" + "="*80)
        print("✅ V10 DATA GENERATION COMPLETE!")
        print("="*80)

        print(f"\n📁 Files created: {self.stats['files_created']}")
        print(f"📊 Total rows: {self.stats['total_rows']:,}")

        gv = self.stats['greeks_validation']
        if gv['total_checks'] > 0:
            print(f"\n🎯 Greeks Validation:")
            print(f"   Deep ITM pass: {gv['deep_itm_passed']}/{gv['total_checks']} ({gv['deep_itm_passed']/gv['total_checks']*100:.1f}%)")
            print(f"   Deep OTM pass: {gv['deep_otm_passed']}/{gv['total_checks']} ({gv['deep_otm_passed']/gv['total_checks']*100:.1f}%)")
            print(f"   ATM pass: {gv['atm_passed']}/{gv['total_checks']} ({gv['atm_passed']/gv['total_checks']*100:.1f}%)")

        if self.stats['quality_issues']:
            print(f"\n⚠️  Quality issues: {len(self.stats['quality_issues'])} (see metadata/quality_issues.txt)")

        print(f"\n📍 Location: {self.base_path}")
        print("="*80)


def main():
    """Generate V10 dataset"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize generator
    generator = V10RealEnhancedGenerator()

    # Generate all data
    stats = generator.generate_all_data()

    return stats


if __name__ == '__main__':
    main()
