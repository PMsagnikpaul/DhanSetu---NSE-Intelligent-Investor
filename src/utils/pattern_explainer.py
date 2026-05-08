# File: src/utils/pattern_explainer.py

import os
from typing import Dict, Optional
from src.config import config


PATTERN_TEMPLATES = {
    'BULLISH_BREAKOUT': (
        "{symbol} has broken above a key resistance level of Rs.{level:.0f} with a "
        "{vol_label} on volume. This typically signals the start of a new upward move. "
        "The stock closed at Rs.{price:.0f}. "
        "Historically, this pattern on {symbol} has worked out {win_rate} of the time "
        "over {samples} similar setups. "
        "Consider: entry near Rs.{entry:.0f}, stop loss at Rs.{stop:.0f}, "
        "target Rs.{target:.0f} ({risk_reward:.1f}x risk-reward). "
        "RSI is at {rsi:.0f} -- {rsi_note}."
    ),
    'BEARISH_BREAKDOWN': (
        "{symbol} has broken below a key support level of Rs.{level:.0f} -- a warning sign. "
        "This may signal continued downside. "
        "Historically on {symbol}, this pattern has preceded further decline {win_rate} of the time. "
        "If you hold {symbol}, review your stop loss. Breakdown level: Rs.{level:.0f}."
    ),
    'HEAD_AND_SHOULDERS': (
        "{symbol} has formed a Head & Shoulders topping pattern -- one of the most reliable "
        "bearish reversal setups. The neckline at Rs.{level:.0f} has been breached. "
        "This historically works {win_rate} of the time on {symbol} ({samples} setups). "
        "Price target based on pattern height: Rs.{target:.0f}."
    ),
    'INV_HEAD_AND_SHOULDERS': (
        "{symbol} has completed an Inverse Head & Shoulders -- a classic bottoming reversal. "
        "The neckline at Rs.{level:.0f} has been cleared. "
        "This pattern has historically signalled a sustained move up in {symbol} "
        "{win_rate} of the time over {samples} setups. Price target: Rs.{target:.0f}."
    ),
    'DOUBLE_TOP': (
        "{symbol} has formed a Double Top -- two failed attempts to break Rs.{level:.0f} "
        "resistance. The neckline has been breached, suggesting the trend may be reversing. "
        "Historically on {symbol}: works {win_rate} of the time over {samples} setups."
    ),
    'DOUBLE_BOTTOM': (
        "{symbol} has formed a Double Bottom at Rs.{level:.0f} -- two successful tests of "
        "support followed by a neckline breakout. A classic bullish reversal. "
        "Historically on {symbol}: works {win_rate} of the time over {samples} setups. "
        "Price target: Rs.{target:.0f}."
    ),
    'SUPPORT_BOUNCE': (
        "{symbol} tested support at Rs.{level:.0f} and bounced. The price held this level "
        "and closed higher -- a short-term positive signal. RSI at {rsi:.0f} confirms "
        "the stock is not yet overbought. Watch for follow-through above Rs.{entry:.0f}."
    ),
    'RESISTANCE_REJECTION': (
        "{symbol} approached resistance at Rs.{level:.0f} but failed to close above it -- "
        "a short-term negative signal. RSI at {rsi:.0f}. "
        "If you hold {symbol}, Rs.{level:.0f} is a key level to monitor."
    ),
}


def _format_win_rate(win_rate_data: Dict) -> tuple:
    """Format win-rate dict into human-readable strings."""
    if not win_rate_data.get('reliable') or win_rate_data.get('win_rate') is None:
        return "an unknown percentage", "insufficient history"
    
    wr  = win_rate_data['win_rate']
    cnt = win_rate_data.get('sample_count', 0)
    return f"{wr * 100:.0f}%", str(cnt)


def generate_template_explanation(pattern: Dict, win_rate_data: Dict) -> str:
    """
    Generate a plain-English explanation using pre-written templates.
    This is the fallback when Claude API is unavailable.
    """
    pt = pattern.get('pattern_type', 'UNKNOWN')
    template = PATTERN_TEMPLATES.get(pt)
    
    if template is None:
        return (
            f"{pattern.get('symbol', '?')} -- {pt.replace('_', ' ').title()} detected. "
            f"Entry: Rs.{pattern.get('entry_price', 0):.0f} | "
            f"Stop: Rs.{pattern.get('stop_loss', 0):.0f} | "
            f"Target: Rs.{pattern.get('price_target', 0):.0f}"
        )
    
    wr_str, samples = _format_win_rate(win_rate_data)
    
    entry  = pattern.get('entry_price', 0)
    stop   = pattern.get('stop_loss', 0)
    target = pattern.get('price_target', 0)
    rsi    = pattern.get('rsi', 50.0)
    
    risk   = abs(entry - stop)
    reward = abs(target - entry)
    rr     = reward / risk if risk > 0 else 0.0
    
    level = (
        pattern.get('breakout_level') or
        pattern.get('breakdown_level') or
        pattern.get('neckline') or
        pattern.get('support_level') or
        pattern.get('resistance_level') or
        pattern.get('top1') or
        pattern.get('bottom1') or
        entry
    )
    
    rsi_note = (
        "healthy momentum" if 40 < rsi < 65 else
        "oversold -- potential bounce" if rsi < 40 else
        "overbought -- be cautious"
    )
    
    vol_label = "strong surge" if pattern.get('volume_ratio', 1.0) >= 1.5 else "average move"
    
    return template.format(
        symbol      = pattern.get('symbol', '?'),
        price       = pattern.get('entry_price', 0),
        entry       = entry,
        stop        = stop,
        target      = target,
        level       = level or entry,
        rsi         = rsi,
        rsi_note    = rsi_note,
        vol_label   = vol_label,
        win_rate    = wr_str,
        samples     = samples,
        risk_reward = rr,
    )


def generate_llm_explanation(pattern: Dict, win_rate_data: Dict) -> str:
    """
    Generate a rich plain-English explanation using Claude API.
    Falls back to template if API key unavailable or call fails.
    """
    api_key = config.ANTHROPIC_API_KEY
    if not api_key:
        return generate_template_explanation(pattern, win_rate_data)
    
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        wr_str, samples = _format_win_rate(win_rate_data)
        
        prompt = f"""You are a financial analyst explaining a stock chart pattern to a first-time Indian retail investor. 
Be clear, concrete, and use simple language. Avoid jargon. Mention key price levels in Indian Rupees (Rs.).

Pattern detected:
- Stock: {pattern.get('symbol')}
- Pattern Type: {pattern.get('pattern_type', '').replace('_', ' ').title()}
- Direction: {pattern.get('direction')}
- Entry Price: Rs.{pattern.get('entry_price', 0):.0f}
- Stop Loss: Rs.{pattern.get('stop_loss', 0):.0f}
- Price Target: Rs.{pattern.get('price_target', 0):.0f}
- Volume Confirmed: {pattern.get('volume_confirmed', False)}
- RSI: {pattern.get('rsi', 50):.0f}

Historical performance of this pattern on this stock:
- Win Rate: {wr_str} (out of {samples} historical setups)
- Average Gain in winning trades: {win_rate_data.get('avg_gain_pct', 'N/A')}%
- Expectancy: {win_rate_data.get('expectancy', 'N/A')}

Write 3 sentences maximum:
1. What the pattern is and what it means in plain English.
2. What the historical data says about this specific pattern on this stock.
3. A clear, actionable statement: what to watch for, what the entry/stop/target are.

Do NOT use bullet points. Do NOT use markdown. Write in plain prose only."""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()
    
    except Exception as e:
        print(f"Warning: Claude API explanation failed: {e}. Using template.")
        return generate_template_explanation(pattern, win_rate_data)