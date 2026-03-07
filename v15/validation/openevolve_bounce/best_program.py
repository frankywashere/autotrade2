import numpy as np

def evaluate_bounce_signal(states: dict, spy_rsi: float) -> dict:
    daily = states.get('daily')
    weekly = states.get('weekly')
    monthly = states.get('monthly')
    if not (daily and weekly):
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    t = 0.40
    if daily['pos_pct'] >= t or weekly['pos_pct'] >= t:
        return {'take_bounce': False, 'delay_hours': 0, 'confidence': 0.0}

    dd, dw = t - daily['pos_pct'], t - weekly['pos_pct']
    c = 0.695 + dd * 1.545 + dw * 0.935
    c += 0.271 * bool(monthly and monthly['pos_pct'] < t)
    c += 0.248 * daily['is_turning'] + 0.335 * weekly['is_turning']
    c += 0.210 * bool(daily.get('at_bottom')) + 0.163 * bool(weekly.get('at_bottom'))
    c += 0.183 * bool(daily.get('near_bottom')) + 0.142 * bool(weekly.get('near_bottom'))
    e = daily.get('energy_ratio', 0.5)
    c += 0.188 if e > 0.64 else -0.135 if e < 0.26 else 0
    c += 0.244 if spy_rsi > 68 else -0.280 if spy_rsi < 30 else 0
    c = np.clip(c, 0.0, 1.0)
    d = 24 if not (daily['is_turning'] or weekly['is_turning']) else (15 if spy_rsi < 30 and not weekly['is_turning'] else 0)

    return {'take_bounce': c >= 0.392, 'delay_hours': d, 'confidence': c}