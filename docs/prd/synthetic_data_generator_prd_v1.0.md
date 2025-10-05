# Synthetic Data Generator PRD v1.0
## Comprehensive Requirements for NIFTY Options Data Generation

### Executive Summary
This PRD defines the requirements for generating synthetic NIFTY options data that supports comprehensive backtesting of options trading strategies, particularly the monthly-weekly coupling strategy. The data must maintain strike persistence, timestamp completeness, and realistic price movements.

---

## 1. CRITICAL DATA INTEGRITY RULES

### 1.1 Strike Price Continuity - MANDATORY
**RULE: Once a strike price exists for an option series, it MUST have continuous data from entry until expiry + 1 day**

```python
Strike_Continuity_Rules = {
    "creation": "When strike appears, it persists until expiry + 1 day",
    "no_gaps": "No missing timestamps during market hours",
    "example": "If strike 24850 exists Monday for July 10 expiry, it MUST have data every 5 minutes through Friday"
}
```

### 1.2 Timestamp Completeness
**Every trading day MUST have complete timestamp coverage**

```python
Required_Timestamps_Per_Day = {
    "5_minute_data": 79,  # 09:15:00 to 15:30:00 in 5-min intervals
    "critical_times": [
        "09:15:00",  # Market open
        "09:30:00",  # Strategy entry time
        "15:15:00",  # Last tradeable time (EXIT ORDERS)
        "15:30:00"   # Settlement time
    ]
}
```

### 1.3 Expiry Day Special Handling
```python
Expiry_Day_Requirements = {
    "09:15-15:15": "Normal trading data every 5 minutes",
    "15:15:00": "MANDATORY - Final tradeable price for exits",
    "15:30:00": "Settlement price (0 for OTM, intrinsic for ITM)",
    "data_availability": "ALL strikes with positions MUST have data"
}
```

---

## 2. DATA SPECIFICATIONS

### 2.1 Five-Minute Data (July 2025 - Present)

#### Instrument Configuration
```python
FIVE_MINUTE_CONFIG = {
    "instrument": "NIFTY",
    "start_date": "2025-07-01",
    "end_date": "current_date",
    "frequency": "5 minutes",
    "timestamps_per_day": 79,
    "trading_hours": {
        "start": "09:15:00",
        "end": "15:30:00"
    }
}
```

#### Expiry Structure Changes
```python
EXPIRY_CHANGES = {
    "before_sept_2025": {
        "weekly": "Thursday",
        "monthly": "Last Thursday"
    },
    "from_sept_2025": {
        "weekly": "Wednesday",  # Changed
        "monthly": "Last Thursday"  # Unchanged
    }
}
```

#### Strike Coverage Requirements
```python
STRIKE_COVERAGE_5MIN = {
    "weekly_options": {
        "range": "ATM ± 1000 points",
        "interval": 50,
        "total_strikes": 41,  # 20 above + ATM + 20 below
        "persistence": "Friday/Monday creation through expiry"
    },
    "monthly_options": {
        "range": "ATM ± 2000 points",
        "interval": 50,
        "total_strikes": 81,  # 40 above + ATM + 40 below
        "persistence": "Month start through month-end expiry"
    }
}
```

### 2.2 Daily Data (January 2023 - Present)

#### Historical Configuration
```python
DAILY_CONFIG = {
    "instrument": "NIFTY",
    "start_date": "2023-01-01",
    "end_date": "current_date",
    "frequency": "Daily",
    "data_point": "End of day (15:30:00)",
    "historical_levels": {
        "2023": {"base": 18000, "range": 1500},
        "2024": {"base": 21500, "range": 2000},
        "2025": {"base": 25000, "range": 2500}
    }
}
```

---

## 3. DATA SCHEMA

### 3.1 Five-Minute Data Fields
```python
FIVE_MINUTE_SCHEMA = {
    "timestamp": "datetime",      # 2025-07-01 09:15:00
    "symbol": "str",              # NIFTY
    "strike": "int",              # 25000, 25050, etc.
    "option_type": "str",         # CE or PE
    "expiry": "date",             # 2025-07-10
    "expiry_type": "str",         # weekly or monthly
    "open": "float",              # NEVER NULL during market
    "high": "float",              
    "low": "float",               
    "close": "float",             # MANDATORY - Never NULL
    "volume": "int",              # Can be 0, never NULL
    "oi": "int",                  
    "bid": "float",               # >= 0.05 for tradeable
    "ask": "float",               
    "iv": "float",                # 0.08 to 0.40 typical
    "delta": "float",             # -1 to 1
    "gamma": "float",             # >= 0
    "theta": "float",             # <= 0
    "vega": "float",              # >= 0
    "underlying_price": "float"   
}
```

### 3.2 Daily Data Fields
```python
DAILY_SCHEMA = {
    "date": "date",               
    "symbol": "str",              
    "strike": "int",              
    "option_type": "str",         
    "expiry": "date",             
    "expiry_type": "str",         
    "open": "float",              
    "high": "float",              
    "low": "float",               
    "close": "float",             
    "settlement_price": "float",  # On expiry day
    "volume": "int",              
    "oi": "int",                  
    "oi_change": "int",           
    "iv": "float",                
    "delta": "float",             
    "gamma": "float",             
    "theta": "float",             
    "vega": "float",              
    "underlying_close": "float",  
    "underlying_high": "float",   
    "underlying_low": "float",    
    "vix_close": "float",         
    "days_to_expiry": "int"       
}
```

---

## 4. STRATEGY SUPPORT REQUIREMENTS

### 4.1 Critical Entry/Exit Points
```python
STRATEGY_CRITICAL_POINTS = {
    "monday_0930": {
        "requirement": "Full weekly option chain",
        "strikes": "ATM ± 1000",
        "data": "All strikes must exist"
    },
    "wednesday_1515": {
        "requirement": "Weekly exit data for DTE=1",
        "critical": "MUST have 15:15:00 timestamp",
        "purpose": "Exit before expiry"
    },
    "thursday_1515": {
        "requirement": "Weekly expiry final prices",
        "settlement": "15:30:00 settlement prices"
    },
    "month_days_1_3": {
        "requirement": "Monthly option chains",
        "strikes": "ATM ± 2000",
        "purpose": "Monthly position entry"
    }
}
```

### 4.2 Coupling Logic Support
```python
COUPLING_REQUIREMENTS = {
    "monthly_exit_trigger": {
        "condition": "When monthly positions exit",
        "requirement": "Weekly strikes MUST have exit prices",
        "timestamp": "Same timestamp as monthly exit"
    },
    "data_synchronization": {
        "rule": "All active positions must have synchronized data",
        "no_orphans": "No position without exit data"
    }
}
```

---

## 5. VALIDATION REQUIREMENTS

### 5.1 Mandatory Validations
```python
def validate_synthetic_data(df, data_type):
    validations = {
        "strike_continuity": check_strike_persistence(df),
        "timestamp_completeness": check_all_timestamps(df),
        "wednesday_1515": check_wednesday_exit_data(df),
        "expiry_settlement": check_expiry_day_data(df),
        "price_sanity": check_price_consistency(df),
        "greek_consistency": check_greek_values(df)
    }
    return all(validations.values())
```

### 5.2 Data Quality Checks
```python
QUALITY_CHECKS = {
    "no_null_closes": "Close price never NULL during market",
    "bid_ask_spread": "Bid <= Price <= Ask always",
    "minimum_price": "No option below 0.05 during trading",
    "otm_decay": "Gradual decay, not sudden drops",
    "itm_intrinsic": "ITM options >= intrinsic value",
    "volume_consistency": "Higher volume near ATM",
    "oi_patterns": "Realistic OI buildup/decay"
}
```

---

## 6. COMMON PITFALLS TO AVOID

```python
AVOID_THESE = {
    "random_strikes": "Don't randomly change available strikes",
    "missing_1515": "Never skip 15:15:00 on any trading day",
    "null_prices": "Never use NULL for close during market",
    "disappearing_strikes": "Don't remove strikes before expiry",
    "zero_before_expiry": "Don't set prices to 0 before 15:30",
    "inconsistent_expiry": "Keep weekly/monthly classification consistent",
    "gaps_in_data": "No gaps in strike data once created"
}
```

---

## 7. IMPLEMENTATION PRIORITIES

### Phase 1: Core Data Generation
1. Strike persistence logic
2. Complete timestamp generation
3. Realistic price movements
4. Greek calculations

### Phase 2: Strategy Support
1. Wednesday 15:15 exit data
2. Coupling synchronization
3. Entry point data completeness
4. Settlement handling

### Phase 3: Validation & Quality
1. Automated validation suite
2. Data quality reports
3. Anomaly detection
4. Performance optimization

---

## 8. SUCCESS CRITERIA

The synthetic data generator will be considered successful when:

1. ✅ 100% strike persistence from creation to expiry
2. ✅ 100% timestamp completeness (79 per day for 5-min)
3. ✅ All Wednesday 15:15:00 timestamps present
4. ✅ All expiry day settlements recorded
5. ✅ Strategy can execute without data gaps
6. ✅ Validation suite passes 100%
7. ✅ Realistic Greeks and price movements
8. ✅ Coupling logic fully supported

---

## 9. TECHNICAL REQUIREMENTS

### Performance
- Generate 1 month of 5-minute data in < 60 seconds
- Generate 1 year of daily data in < 30 seconds
- Memory usage < 2GB for 1 year of data

### Output Format
- CSV files with proper headers
- Organized by date/month folders
- Compression supported for storage

### Compatibility
- Python 3.8+
- Pandas DataFrame compatible
- Standard datetime formats
- Float precision: 2 decimal places for prices

---

## 10. APPENDIX: Market Holidays

NSE holidays must be excluded from data generation:
- Republic Day (Jan 26)
- Holi (varies)
- Good Friday (varies)
- Ambedkar Jayanti (Apr 14)
- Maharashtra Day (May 1)
- Independence Day (Aug 15)
- Gandhi Jayanti (Oct 2)
- Diwali (varies)
- Guru Nanak Jayanti (varies)

---

*End of PRD v1.0*