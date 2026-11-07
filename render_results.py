from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
FIGURES = ROOT/'figures'
FIGURES.mkdir(exist_ok=True)
trials = pd.read_csv(RESULTS/'followup_trials.csv')
pairs = pd.read_csv(RESULTS/'followup_pairs.csv')
description = pd.read_csv(RESULTS/'followup_descriptive_pairs.csv')
sweep = pd.read_csv(RESULTS/'followup_sweep_pairs.csv')
controls = pd.read_csv(RESULTS/'followup_control_pairs.csv')
fields = pd.read_csv(RESULTS/'followup_power_fields.csv')
fits = pd.read_csv(RESULTS/'followup_sweep_fits.csv')
aa = pd.read_csv(RESULTS/'followup_aa_pairs.csv')
audit = json.loads((RESULTS/'followup_audit.json').read_text())
scenario_names = {'static': 'Static', 'sparse_changes': 'Sparse', 'all_changed': 'All changed'}
policy_names = {'full_render': 'Full', 'validate_reuse': 'Validate', 'known_dirty': 'Dirty', 'known_compact': 'Compact', 'no_launch': 'Idle', 'copy_only': 'Copy'}
scenario_order = ['static', 'sparse_changes', 'all_changed']
policy_order = ['full_render', 'validate_reuse', 'known_dirty', 'known_compact']


def table(name, columns, header, rows):
    text = '\\begin{tabular}{'+columns+'}\n\\toprule\n'
    text += ' & '.join(header)+' \\\\\n\\midrule\n'
    text += ''.join(' & '.join(map(str, row))+' \\\\\n' for row in rows)
    text += '\\bottomrule\n\\end{tabular}\n'
    (RESULTS/name).write_text(text)


def span(values, digits=2):
    return f'{values.median():.{digits}f} [{values.min():.{digits}f}, {values.max():.{digits}f}]'


def stored_span(row, column):
    return f'{row[column+"_median"]:.2f} [{row[column+"_min"]:.2f}, {row[column+"_max"]:.2f}]'


table('protocol_table.tex', 'lrr', ['Component', 'Trials', 'Window (s)'], [
    ['Policy comparison', 48, 20], ['Polling-rate sweep', 52, 15],
    ['Idle/copy controls', 24, 20], ['Identical-code A/A', 8, 20],
    ['Explicit power fields', 8, 20], ['Sustained load', 4, 15], ['Total', 144, '--'],
])
rows = []
for scenario in scenario_order:
    for i, policy in enumerate(policy_order[1:]):
        row = description[(description.scenario == scenario) & (description.policy == policy)].iloc[0]
        rows.append([scenario_names[scenario] if i == 0 else '', policy_names[policy],
                     stored_span(row, 'counter_saving_pct'), stored_span(row, 'power_saving_pct'),
                     f'{int(row.opposite_signs)}/4'])
table('energy_table.tex', 'llrrc', ['Scenario', 'Policy', 'Counter saving (\\%)', 'Power-integral saving (\\%)', 'Opposite'], rows)
rows = []
timing_rows = []
for scenario in scenario_order:
    for i, policy in enumerate(policy_order):
        group = trials[(trials.phase == 'main') & (trials.scenario == scenario) & (trials.policy == policy)]
        wall = group.wall_ms_median
        rows.append([scenario_names[scenario] if i == 0 else '', policy_names[policy],
                     f'{wall.median():.3f} [{wall.quantile(.25):.3f}, {wall.quantile(.75):.3f}]',
                     f'{group.gpu_ms_median.median():.3f}', f'{group.render_ms_median.median():.3f}',
                     f'{group.copy_ms_median.median():.3f}', f'{group.power_w.median():.3f}'])
        detail = {'scenario': scenario, 'policy': policy, 'trials': len(group)}
        for column in ['wall_ms', 'gpu_ms', 'upload_ms', 'validate_ms', 'render_ms', 'copy_ms', 'controller_ms']:
            value = group[column+'_median']
            detail.update({column+'_median': value.median(), column+'_q25': value.quantile(.25), column+'_q75': value.quantile(.75)})
        timing_rows.append(detail)
table('timing_table.tex', 'llrrrrr', ['Scenario', 'Policy', 'Wall time [Q1, Q3] (ms)', 'GPU (ms)', 'Render (ms)', 'Copy (ms)', '$\\widehat P$ (W)'], rows)
pd.DataFrame(timing_rows).to_csv(RESULTS/'timing_components.csv', index=False)
rows = []
for interval, group in sweep.groupby('interval_ms'):
    rows.append([interval, f'{group.actual_counter_mode_poll_ms.median():.1f}',
                 f'{group.power_mode_ratio.median():.2f}', f'{group.counter_mode_ratio.median():.2f}'])
table('polling_table.tex', 'rrrr', ['Nominal (ms)', 'Achieved (ms)', 'P: $C/P$', 'P+C: $C/P$'], rows)
rows = []
selected = [('Full, static', 'P', trials[(trials.phase=='main') & (trials.scenario=='static') & (trials.policy=='full_render')])]
for policy in ['no_launch', 'copy_only']:
    for mode, name in [('power','P'), ('power_counter','P+C'), ('boundary_only','B')]:
        selected.append((policy_names[policy], name, trials[(trials.phase=='controls') & (trials.policy==policy) & (trials.monitor==mode)]))
selected.append(('Sustained', 'P', trials[trials.phase=='sustained_load']))
for label, mode, group in selected:
    rows.append([label, mode, span(group.counter_w), span(group.power_w) if group.power_w.notna().any() else '--',
                 span(group.ratio) if group.ratio.notna().any() else '--'])
table('controls_table.tex', 'llrrr', ['Workload', 'Mode', '$\\widehat E^C/\\Delta t$ (W)', '$\\widehat E^P/\\Delta t$ (W)', '$\\widehat E^C/\\widehat E^P$'], rows)
rows = [[int(row.session)+1, f'{row.counter_B_vs_A_pct:.2f}', f'{row.power_B_vs_A_pct:.2f}'] for row in aa.itertuples()]
table('aa_table.tex', 'crr', ['Session', 'Counter saving (\\%)', 'Power saving (\\%)'], rows)
rows = []
for _, row in fields.iterrows():
    rows.append([int(row.session)+1, f'{row["185_vs_direct_pct"]:.3f}', f'{row["186_vs_direct_pct"]:.3f}'])
table('fields_table.tex', 'crr', ['Session', 'Average field (\\%)', 'Instant field (\\%)'], rows)

macros = {
    'FollowupTrials': str(audit['completed_trials']), 'FollowupFrames': f'{audit["measured_frames"]:,}',
    'PacedFrames': f'{audit["paced_frames"]:,}', 'FollowupChecks': f'{audit["correctness_checks"]:,}',
    'FollowupPixels': f'{audit["correctness_pixels"]:,}', 'FollowupRuntime': f'{audit["elapsed_wall_s"]/60:.2f}',
    'FollowupOpposite': str(audit['opposite_signs']), 'CounterRatioFast': f'{sweep[sweep.interval_ms==25].counter_mode_ratio.median():.2f}',
    'CounterRatioSlow': f'{sweep[sweep.interval_ms==1000].counter_mode_ratio.median():.2f}',
    'IdleRatioMin': f'{controls[controls.policy=="no_launch"].counter_mode_ratio.min():.2f}',
    'IdleRatioMax': f'{controls[controls.policy=="no_launch"].counter_mode_ratio.max():.2f}',
    'CounterSlopeMin': f'{fits.slope_reported_j_per_read.min():.2f}',
    'CounterSlopeMax': f'{fits.slope_reported_j_per_read.max():.2f}',
    'CounterAaSd': f'{aa.counter_B_vs_A_pct.std():.2f}', 'PowerAaSd': f'{aa.power_B_vs_A_pct.std():.2f}',
}
(RESULTS/'numbers.tex').write_text(''.join('\\newcommand{\\'+name+'}{'+value+'}\n' for name,value in macros.items()))
(RESULTS/'paper_numbers.json').write_text(json.dumps(macros, indent=2)+'\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
colors = ['#1f77b4', '#b04b32', '#2b8c5a', '#8b65a5']
fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.65))
for session, group in sweep.groupby('session'):
    group = group.sort_values('interval_ms')
    axes[0].plot(group.interval_ms, group.counter_mode_ratio, 'o-', color=colors[int(session)],
                 ms=3.1, lw=.9, label=f'Session {int(session)+1}')
    axes[0].plot(group.interval_ms, group.power_mode_ratio, 'x--', color=colors[int(session)], ms=3.3, lw=.6, alpha=.65)
axes[0].axhline(1, color='black', linewidth=.6)
axes[0].set(xscale='log', yscale='log', xlabel='Nominal polling interval (ms)',
            ylabel='Counter / integrated power', title='(a) Static full-render polling sweep')
axes[0].set_xticks([25,50,100,250,500,1000], ['25','50','100','250','500','1000'])
axes[0].set_yticks([1,3,10,30,100], ['1','3','10','30','100'])
axes[0].legend(loc='upper right', frameon=False, fontsize=6.5)
axes[0].text(.04,.30,'Solid: P+C   Dashed: P', transform=axes[0].transAxes, fontsize=7)
markers = {'validate_reuse': 'o', 'known_dirty': 's', 'known_compact': '^'}
scenario_colors = {'static': colors[0], 'sparse_changes': colors[1], 'all_changed': colors[2]}
for (scenario, policy), group in pairs.groupby(['scenario','policy']):
    axes[1].scatter(group.power_saving_pct, group.counter_saving_pct, marker=markers[policy],
                    color=scenario_colors[scenario], s=17, alpha=.85, edgecolors='white', linewidths=.3)
axes[1].axhline(0, color='black', linewidth=.7)
axes[1].axvline(0, color='black', linewidth=.7)
axes[1].plot([-5,25], [-5,25], color='gray', linewidth=.6, linestyle=':')
axes[1].set(xlabel='Power-integral saving (%)', ylabel='Counter saving (%)',
            title='(b) Matched policy comparisons', xlim=(-4,15), ylim=(-31,27))
legend = [Line2D([0],[0],marker='o',linestyle='',color=scenario_colors[s],label=scenario_names[s],markersize=4) for s in scenario_order]
axes[1].legend(handles=legend, frameon=False, fontsize=6.5, loc='lower right')
fig.tight_layout(w_pad=2)
fig.savefig(FIGURES/'polling_and_pairs.pdf', bbox_inches='tight')
fig.savefig(FIGURES/'polling_and_pairs.png', dpi=220, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.7,3.4))
for session, group in sweep.groupby('session'):
    group = group.sort_values('counter_mode_reads')
    fit = fits[fits.session==session].iloc[0]
    ax.plot(group.counter_mode_reads, group.reported_extra_counter_j, 'o', ms=4, color=colors[int(session)], label=f'Session {int(session)+1}')
    xx = np.linspace(group.counter_mode_reads.min(),group.counter_mode_reads.max(),80)
    ax.plot(xx, fit.slope_reported_j_per_read*xx+fit.intercept_j, '-', color=colors[int(session)], lw=.8)
ax.set(xlabel='Continuous counter reads inside the counter window', ylabel='Additional reported counter energy (J)',
       title='P+C minus matched P trial; descriptive fits')
ax.legend(frameon=False);fig.tight_layout()
fig.savefig(FIGURES/'counter_read_count.pdf')
fig.savefig(FIGURES/'counter_read_count.png', dpi=200)
plt.close(fig)
print('Tables, scalar values, and figures regenerated from audited trial tables.')
