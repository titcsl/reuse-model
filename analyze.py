import argparse
import base64
import bisect
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import zlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASETS = {
    'pilot': ('gpu_audit_v2_20261102_144917_80abb6.csv', 'ebd9cd9b636b363d68f4d7129432da55828c7b27773352c45fd998b62c1e2fff'),
    'followup': ('gpu_audit_v3_20261106_153155_70b1e6.csv', 'd88e799c18ac7f6122e2b6192fa7bdfb68b634dc46805272dd37b1623041efee'),
}
POLICIES = ['full_render', 'validate_reuse', 'known_dirty', 'known_compact']
TIMING_COLUMNS = ['wall_ms', 'gpu_ms', 'upload_ms', 'validate_ms', 'render_ms', 'copy_ms', 'controller_ms']


def midpoint(query):
    return (query['before_ns'] + query['after_ns']) / 2


def integrate(points, start, end):
    points = sorted(points)
    if len(points) < 2 or start < points[0][0] or end > points[-1][0] or end <= start:
        return None
    times = np.array([point[0] - start for point in points], dtype=float) / 1e9
    power = np.array([point[1] for point in points], dtype=float)
    duration = (end - start) / 1e9
    mask = (times > 0) & (times < duration)
    clipped_times = np.r_[0., times[mask], duration]
    clipped_power = np.interp(clipped_times, times, power)
    return float(np.sum(np.diff(clipped_times) * (clipped_power[:-1] + clipped_power[1:]) / 2))


def summary(values):
    numbers = pd.Series(values, dtype=float).dropna()
    if numbers.empty:
        return {'n': 0}
    return {
        'n': len(numbers), 'median': float(numbers.median()),
        'min': float(numbers.min()), 'max': float(numbers.max()),
        'q25': float(numbers.quantile(.25)), 'q75': float(numbers.quantile(.75)),
        'sd': float(numbers.std(ddof=1)) if len(numbers) > 1 else None,
    }


def read_acquisition(path, expected_hash, label):
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f'Input checksum mismatch: {path.name}')
    trials, plans = {}, {}
    telemetry, frames, boundaries = (collections.defaultdict(list) for _ in range(3))
    metadata, statuses, correctness, negative, final_images = [], [], [], [], []
    counts = collections.Counter()
    problems = []
    with path.open(newline='', encoding='utf-8-sig') as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            if None in row:
                problems.append(f'Extra CSV fields on line {line_number}')
            kind = row['record_type']
            counts[kind] += 1
            key = (row['session'], row['trial'])
            if kind in ('trial', 'trial_plan'):
                destination = trials if kind == 'trial' else plans
                if key in destination:
                    problems.append(f'Duplicate {kind}: {key}')
                destination[key] = (row, json.loads(row['details']))
            elif kind == 'frame':
                value = {name: float(row[name]) if row[name] else None for name in TIMING_COLUMNS + ['dirty_fraction', 'lateness_ms']}
                value.update(frame=int(row['frame']), start_ns=int(row['t_start_ns']), end_ns=int(row['t_end_ns']),
                             deadline_miss=int(row['deadline_miss']) if row['deadline_miss'] else None)
                frames[key].append(value)
            elif kind == 'telemetry':
                telemetry[key].append(json.loads(row['details']))
            elif kind == 'energy_boundary':
                boundaries[key].append({
                    'name': row['metric'], 'before_ns': int(row['t_start_ns']), 'after_ns': int(row['t_end_ns']),
                    'value': int(row['value']) if row['value'] else None, 'status': json.loads(row['details'])['status'],
                })
            elif kind == 'metadata':
                metadata.append(row)
            elif kind in ('run_status', 'session_status', 'run_error', 'session_transition'):
                statuses.append(row)
            elif kind in ('correctness', 'negative_control', 'final_image_check'):
                {'correctness': correctness, 'negative_control': negative, 'final_image_check': final_images}[kind].append(row)
    top = {row['metric']: row['value'] for row in metadata if row['session'] == ''}
    source = zlib.decompress(base64.b64decode(top['payload_source_zlib_base64']))
    source_hash = hashlib.sha256(source).hexdigest()
    if source_hash != top['payload_sha256']:
        problems.append('Embedded source checksum mismatch')
    protocol = json.loads(top['protocol'])
    compact = json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if hashlib.sha256(compact.encode()).hexdigest() != top['protocol_sha256']:
        problems.append('Protocol checksum mismatch')
    if label == 'followup':
        for session, planned in enumerate(protocol['plan']):
            for index, expected in enumerate(planned, 1):
                key = (str(session), str(index))
                if key not in plans:
                    problems.append(f'Missing planned trial: {key}')
                    continue
                actual = plans[key][1]
                if any(actual.get(name) != value for name, value in expected.items()):
                    problems.append(f'Acquisition diverged from fixed plan: {key}')
    trial_rows, field_rows, query_rows, phase_rows = [], [], [], []
    operating = collections.Counter()
    maximum_integration_error = 0.
    for key, (row, details) in trials.items():
        if details['status'] != 'completed':
            continue
        plan = plans[key][1]
        start, end = int(row['t_start_ns']), int(row['t_end_ns'])
        start_counter = next(value for value in boundaries[key] if value['name'] == 'start')
        end_counter = next(value for value in boundaries[key] if value['name'] == 'end')
        if start_counter['status'] != 0 or end_counter['status'] != 0:
            problems.append(f'Counter boundary status: {key}')
        counter_j = (end_counter['value'] - start_counter['value']) / 1000
        if not math.isclose(counter_j, float(row['energy_j']), abs_tol=1e-9):
            problems.append(f'Counter total mismatch: {key}')
        if counter_j < 0:
            problems.append(f'Negative counter difference: {key}')
        point_list = [(midpoint(value['power']), value['power']['value'] / 1000)
                      for value in telemetry[key] if value['power']['status'] == 0]
        counter_start, counter_end = midpoint(start_counter), midpoint(end_counter)
        power_j = integrate(point_list, counter_start, counter_end)
        work_j = integrate(point_list, start, end)
        for calculated, field in [(power_j, 'power_integral_counter_midpoints_j'), (work_j, 'power_integral_work_j')]:
            recorded = details[field]
            if (calculated is None) != (recorded is None):
                problems.append(f'Integral availability mismatch: {key}, {field}')
            elif calculated is not None:
                error = abs(calculated - recorded)
                maximum_integration_error = max(maximum_integration_error, error)
                if error > 1e-7:
                    problems.append(f'Integral mismatch: {key}, {field}: {error}')
        trial_frames = frames[key]
        if [value['frame'] for value in trial_frames] != list(range(details['completed_frames'])):
            problems.append(f'Frame sequence/count mismatch: {key}')
        paced = details.get('paced', True)
        if paced and len(trial_frames) != round(plan['seconds'] * 60):
            problems.append(f'Incomplete paced trial: {key}')
        if abs((end - start) / 1e9 - details['duration_s']) > 1e-9:
            problems.append(f'Work-duration mismatch: {key}')
        for name in ('block', 'phase', 'condition', 'scenario', 'policy'):
            if str(plan[name]) != row[name]:
                problems.append(f'Trial-plan mismatch: {key}, {name}')
        result = {name: row[name] for name in ('session', 'trial', 'block', 'phase', 'condition', 'scenario', 'policy', 'seed')}
        result.update(dataset=label, monitor=plan['monitor'], nominal_poll_ms=plan.get('interval_ms', 100),
                      fields_enabled=plan.get('fields', False), width=plan.get('width', 960), height=plan.get('height', 540),
                      paced=paced, start_ns=start, end_ns=end, duration_s=(end-start)/1e9,
                      planned_duration_s=plan['seconds'], counter_duration_s=(counter_end-counter_start)/1e9,
                      frames=len(trial_frames), counter_j=counter_j, power_j=power_j, power_work_j=work_j,
                      counter_w=counter_j/((counter_end-counter_start)/1e9),
                      power_w=power_j/((counter_end-counter_start)/1e9) if power_j is not None else None,
                      power_j_per_frame=work_j/len(trial_frames) if work_j is not None else None,
                      ratio=counter_j/power_j if power_j else None,
                      deadline_misses=sum(value['deadline_miss'] or 0 for value in trial_frames),
                      dirty_fraction=statistics.mean(value['dirty_fraction'] for value in trial_frames),
                      state_trace_sha256=details['state_trace_sha256'], warm_trace_sha256=details['warm_trace_sha256'],
                      cpu_process_s=details['cpu_process_s'], flags=';'.join(details['flags']),
                      ac_start=details['ac_start'], ac_end=details['ac_end'])
        for name in TIMING_COLUMNS:
            values = [value[name] for value in trial_frames if value[name] is not None]
            for metric, value in summary(values).items():
                result[name+'_'+metric] = value
        if result['deadline_misses'] != details['deadline_misses']:
            problems.append(f'Deadline mismatch: {key}')
        if not math.isclose(result['wall_ms_median'], details['wall_p50_ms'], abs_tol=1e-9):
            problems.append(f'Wall median mismatch: {key}')
        frame_starts = [value['start_ns'] for value in trial_frames]
        frame_ends = [value['end_ns'] for value in trial_frames]
        observed_power_times = []
        local_counter = []
        states = collections.Counter()
        for snapshot in telemetry[key]:
            query = snapshot['power']
            stamp = midpoint(query)
            operating[snapshot['pstate'], snapshot['sm_mhz']] += 1
            if query['status'] != 0:
                problems.append(f'Power-query status: {key}')
            in_window = start <= stamp <= end
            state_query = snapshot.get('pstate_query', query)
            state_stamp = midpoint(state_query)
            if start <= state_stamp <= end:
                states[snapshot['pstate'], snapshot['sm_mhz']] += 1
            if in_window:
                observed_power_times.append(stamp)
                position = bisect.bisect_right(frame_starts, stamp) - 1
                phase_rows.append({
                    'dataset': label, 'session': key[0], 'trial': key[1], 'phase': row['phase'],
                    'poll_phase_ms': ((stamp-start)/1e6) % (1000/60),
                    'power_query_during_host_frame': int(position >= 0 and stamp <= frame_ends[position]),
                    'pstate': snapshot['pstate'], 'sm_mhz': snapshot['sm_mhz'],
                })
            counter = snapshot.get('counter')
            if counter and counter['status'] == 0:
                local_counter.append((midpoint(counter), counter['value'], 'continuous', counter['after_ns']-counter['before_ns']))
            for field in snapshot.get('power_fields', []):
                field_rows.append({
                    'session': key[0], 'trial': key[1], 'in_work_window': in_window,
                    'query_ns': midpoint(field), 'direct_mw': query['value'],
                    'wall_time_ns': snapshot['sample_wall_time_ns'], **field,
                })
        for boundary in boundaries[key]:
            if boundary['status'] == 0:
                local_counter.append((midpoint(boundary), boundary['value'], boundary['name'], boundary['after_ns']-boundary['before_ns']))
        local_counter.sort()
        previous = None
        for stamp, value, kind, latency in local_counter:
            if previous is not None and value < previous[1]:
                problems.append(f'Nonmonotonic local counter: {key}')
            query_rows.append({
                'dataset': label, 'session': key[0], 'trial': key[1], 'phase': row['phase'],
                'monitor': plan['monitor'], 'policy': row['policy'], 'query_kind': kind,
                'nominal_poll_ms': plan.get('interval_ms', 100), 'time_ns': stamp,
                'relative_s': (stamp-start)/1e9, 'value_mj': value, 'latency_ms': latency/1e6,
                'in_work_window': start <= stamp <= end, 'in_counter_window': counter_start <= stamp <= counter_end,
                'delta_j': (value-previous[1])/1000 if previous is not None else None,
                'delta_s': (stamp-previous[0])/1e9 if previous is not None else None,
            })
            previous = (stamp, value)
        result['continuous_counter_queries'] = sum(kind == 'continuous' and counter_start < stamp < counter_end for stamp, value, kind, latency in local_counter)
        result['power_queries_in_work'] = len(observed_power_times)
        result['state_queries_in_work'] = sum(states.values())
        intervals = np.diff(observed_power_times)/1e6
        result['actual_poll_ms_median'] = float(np.median(intervals)) if len(intervals) else None
        result['actual_poll_ms_max'] = float(np.max(intervals)) if len(intervals) else None
        result['p8_fraction_in_work'] = sum(n for (pstate, clock), n in states.items() if pstate == 8)/sum(states.values()) if states else None
        result['p0_fraction_in_work'] = sum(n for (pstate, clock), n in states.items() if pstate == 0)/sum(states.values()) if states else None
        result['temperature_c_max'] = max(value['temperature_c'] for value in telemetry[key]) if telemetry[key] else None
        result['pstates_in_work'] = ','.join(str(value) for value in sorted({pair[0] for pair in states}))
        trial_rows.append(result)
    for row in correctness + final_images:
        if int(row['value']) != 0:
            problems.append(f'Correctness mismatch: {row["record_type"]}, {row["session"]}, {row["trial"]}')
    if len(final_images) != len(trial_rows):
        problems.append('Final-image count differs from completed-trial count')
    if not any(int(row['value']) > 0 for row in negative):
        problems.append('Missing detecting negative control')
    valid_pixels = sum(json.loads(row['details'])['width'] * json.loads(row['details'])['height'] for row in correctness)
    run_states = [row for row in statuses if row['record_type'] == 'run_status']
    elapsed = (datetime.datetime.fromisoformat(run_states[-1]['utc'])-datetime.datetime.fromisoformat(run_states[0]['utc'])).total_seconds()
    audit = {
        'dataset': label, 'input_file': path.name, 'input_sha256': actual_hash, 'source_sha256': source_hash,
        'protocol_sha256': top['protocol_sha256'], 'planned_trials': protocol['planned_trials'],
        'completed_trials': len(trial_rows), 'record_counts': dict(counts), 'run_status': run_states[-1]['metric'],
        'elapsed_wall_s': elapsed, 'correctness_checks': len(correctness), 'correctness_pixels': valid_pixels,
        'negative_control_mismatches': sum(int(row['value']) for row in negative),
        'final_image_checks': len(final_images), 'maximum_integral_recalculation_error_j': maximum_integration_error,
        'operating_states_all_telemetry': [{'pstate': k[0], 'sm_mhz': k[1], 'count': v} for k,v in operating.most_common()],
        'problems': problems,
    }
    return {
        'trials': pd.DataFrame(trial_rows), 'fields': pd.DataFrame(field_rows),
        'queries': pd.DataFrame(query_rows), 'phases': pd.DataFrame(phase_rows),
        'audit': audit, 'metadata': metadata, 'protocol': protocol, 'source': source,
    }


def main_pairs(trials):
    rows = []
    for (session, block, scenario), group in trials[trials.phase == 'main'].groupby(['session', 'block', 'scenario']):
        if len(group) != 4 or set(group.policy) != set(POLICIES):
            continue
        if len(set(group.state_trace_sha256)) != 1 or len(set(group.warm_trace_sha256)) != 1:
            raise ValueError('Matched policies consumed different traces')
        baseline = group[group.policy == 'full_render'].iloc[0]
        for _, trial in group[group.policy != 'full_render'].iterrows():
            counter = 100 * (1-trial.counter_j/baseline.counter_j)
            power = 100 * (1-trial.power_j/baseline.power_j)
            rows.append({
                'session': session, 'block': block, 'scenario': scenario, 'policy': trial.policy,
                'full_trial': baseline.trial, 'reuse_trial': trial.trial,
                'counter_saving_pct': counter, 'power_saving_pct': power,
                'sign_disagreement': counter*power < 0, 'power_difference_j': baseline.power_j-trial.power_j,
                'power_difference_w': baseline.power_w-trial.power_w,
                'wall_reduction_pct': 100*(1-trial.wall_ms_median/baseline.wall_ms_median),
                'full_misses': baseline.deadline_misses, 'reuse_misses': trial.deadline_misses,
            })
    return pd.DataFrame(rows)


def describe_pairs(pairs):
    rows = []
    for (scenario, policy), group in pairs.groupby(['scenario', 'policy']):
        result = {'scenario': scenario, 'policy': policy, 'blocks': len(group),
                  'opposite_signs': int(group.sign_disagreement.sum())}
        for column in ('counter_saving_pct', 'power_saving_pct', 'power_difference_w', 'wall_reduction_pct'):
            for metric, value in summary(group[column]).items():
                result[column+'_'+metric] = value
        rows.append(result)
    return pd.DataFrame(rows)


def aa_pairs(trials):
    rows = []
    for (session, block), group in trials[trials.phase == 'aa'].groupby(['session', 'block']):
        if len(group) != 2:
            continue
        a, b = (group[group.condition == label].iloc[0] for label in ('A', 'B'))
        rows.append({'session': session, 'block': block,
                     'counter_B_vs_A_pct': 100*(1-b.counter_j/a.counter_j),
                     'power_B_vs_A_pct': 100*(1-b.power_j/a.power_j),
                     'start_separation_s': abs(b.start_ns-a.start_ns)/1e9})
    return pd.DataFrame(rows)


def followup_diagnostics(data):
    trials, fields, queries = data['trials'], data['fields'], data['queries']
    sweep = []
    for (session, interval), group in trials[trials.phase == 'poll_sweep'].groupby(['session', 'nominal_poll_ms']):
        if not {'power', 'power_counter'}.issubset(set(group.monitor)):
            continue
        power = group[group.monitor == 'power'].iloc[0]
        both = group[group.monitor == 'power_counter'].iloc[0]
        sweep.append({
            'session': session, 'interval_ms': interval,
            'power_mode_counter_j': power.counter_j, 'counter_mode_counter_j': both.counter_j,
            'power_mode_power_j': power.power_j, 'counter_mode_power_j': both.power_j,
            'power_mode_ratio': power.ratio, 'counter_mode_ratio': both.ratio,
            'counter_mode_reads': both.continuous_counter_queries,
            'actual_counter_mode_poll_ms': both.actual_poll_ms_median,
            'reported_extra_counter_j': both.counter_j-power.counter_j,
            'reported_extra_j_per_read': (both.counter_j-power.counter_j)/both.continuous_counter_queries,
            'integral_change_pct': 100*(both.power_j/power.power_j-1),
        })
    sweep = pd.DataFrame(sweep)
    fits = []
    for session, group in sweep.groupby('session'):
        x = group.counter_mode_reads.to_numpy(float)
        y = group.reported_extra_counter_j.to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        residual = y-(slope*x+intercept)
        fits.append({'session': session, 'points': len(group), 'slope_reported_j_per_read': slope,
                     'intercept_j': intercept, 'r_squared': 1-np.sum(residual**2)/np.sum((y-y.mean())**2)})
    field_summary = []
    for (session, trial), group in fields.groupby(['session', 'trial']):
        selected = trials[(trials.session == session) & (trials.trial == trial)].iloc[0]
        result = {'session': session, 'trial': trial, 'direct_integral_work_j': selected.power_work_j}
        for identifier in (185, 186):
            part = group[(group.field_id == identifier) & (group.api_status == 0) & (group.field_status == 0)]
            points = list(zip(part.query_ns, part.value_mw/1000))
            value = integrate(points, selected.start_ns, selected.end_ns)
            result[str(identifier)+'_integral_work_j'] = value
            result[str(identifier)+'_vs_direct_pct'] = 100*(value/selected.power_work_j-1)
            measured = part[part.in_work_window]
            result[str(identifier)+'_source_timestamp_minus_snapshot_wall_ms_median'] = float(np.median(measured.source_timestamp_us/1000-measured.wall_time_ns/1e6))
        field_summary.append(result)
    controls = []
    for (session, policy), group in trials[trials.phase == 'controls'].groupby(['session', 'policy']):
        p = group[group.monitor == 'power'].iloc[0]
        pc = group[group.monitor == 'power_counter'].iloc[0]
        b = group[group.monitor == 'boundary_only'].iloc[0]
        controls.append({'session': session, 'policy': policy, 'power_mode_w': p.power_w,
                         'counter_mode_integral_w': pc.power_w, 'counter_mode_implied_w': pc.counter_w,
                         'power_mode_counter_w': p.counter_w, 'boundary_counter_w': b.counter_w,
                         'counter_mode_ratio': pc.ratio, 'power_mode_ratio': p.ratio,
                         'integral_change_pct': 100*(pc.power_j/p.power_j-1),
                         'counter_mode_reads': pc.continuous_counter_queries})
    return {'sweep_pairs': sweep, 'sweep_fits': pd.DataFrame(fits),
            'power_fields': pd.DataFrame(field_summary), 'control_pairs': pd.DataFrame(controls)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.root/'results'
    output.mkdir(exist_ok=True, parents=True)
    combined = {}
    for label, (name, expected_hash) in DATASETS.items():
        data = read_acquisition(args.root/'data'/name, expected_hash, label)
        pairs = main_pairs(data['trials'])
        data['pairs'] = pairs
        data['descriptive_pairs'] = describe_pairs(pairs)
        data['aa_pairs'] = aa_pairs(data['trials'])
        for name in ('trials', 'pairs', 'descriptive_pairs', 'aa_pairs', 'fields', 'queries', 'phases'):
            data[name].to_csv(output/(label+'_'+name+'.csv'), index=False)
        if label == 'followup':
            diagnostics = followup_diagnostics(data)
            for name, frame in diagnostics.items():
                frame.to_csv(output/('followup_'+name+'.csv'), index=False)
            data['diagnostics'] = diagnostics
        audit = data['audit']
        audit.update(main_blocks=int(pairs.groupby(['session', 'block']).ngroups), main_contrasts=len(pairs),
                     opposite_signs=int(pairs.sign_disagreement.sum()),
                     opposite_signs_both_over_1pct=int((pairs.sign_disagreement & (pairs.counter_saving_pct.abs()>1) & (pairs.power_saving_pct.abs()>1)).sum()),
                     opposite_signs_both_over_5pct=int((pairs.sign_disagreement & (pairs.counter_saving_pct.abs()>5) & (pairs.power_saving_pct.abs()>5)).sum()),
                     measured_frames=int(data['trials'].frames.sum()),
                     paced_frames=int(data['trials'].loc[data['trials'].paced, 'frames'].sum()),
                     deadline_misses=int(data['trials'].deadline_misses.sum()),
                     maximum_duration_error_s=float((data['trials'].duration_s-data['trials'].planned_duration_s).abs().max()),
                     temperature_c=summary(data['trials'].temperature_c_max))
        (output/(label+'_audit.json')).write_text(json.dumps(audit, indent=2)+'\n')
        (output/(label+'_protocol.json')).write_text(json.dumps(data['protocol'], indent=2)+'\n')
        clean_metadata = [{key: row[key] for key in ('session', 'metric', 'value')}
                          for row in data['metadata'] if row['metric'] != 'payload_source_zlib_base64']
        (output/(label+'_metadata.json')).write_text(json.dumps(clean_metadata, indent=2)+'\n')
        if audit['problems']:
            raise ValueError(audit['problems'])
        combined[label] = audit
        print(json.dumps(audit, indent=2))
    (output/'audit_summary.json').write_text(json.dumps(combined, indent=2)+'\n')


if __name__ == '__main__':
    main()
