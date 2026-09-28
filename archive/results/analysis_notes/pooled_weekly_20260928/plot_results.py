"""Static research charts from saved experiment outputs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
p3 = pd.read_csv(ROOT/'archive/results/model_development/20260928_p3_01/metrics.csv')
p4 = pd.read_csv(ROOT/'archive/results/model_development/20260928_p4_01/metrics.csv')
paths = pd.read_csv(HERE/'current_paths.csv',parse_dates=['target_month'])
fig, axs = plt.subplots(1,3,figsize=(17,5))
p3.pivot(index='model',columns='horizon',values='MAE_common').plot.barh(ax=axs[0])
axs[0].set(title='P3: monthly MAE (Jan 2024–Aug 2026)',xlabel='Percentage points',ylabel='')
p4.pivot(index='stage',columns='model',values='MAE_common').plot(marker='o',ax=axs[1])
axs[1].set(title='P4: MAE (paired N = 16 / 16 / 16 / 15)',xlabel='Observed weekly dates',ylabel='Percentage points',xticks=[1,2,3,4])
paths.pivot(index='target_month',columns='model',values='prediction').drop(columns=['SeasonalNaive','GroupsFixed']).plot(ax=axs[2])
axs[2].set(title='Experimental monthly paths (no tariff override)',xlabel='',ylabel='MoM, %')
fig.suptitle('Historical revised-data diagnostics (C); no production promotion',fontsize=13)
fig.tight_layout()
fig.savefig(HERE/'comparison.png',dpi=150)
print(HERE/'comparison.png')
