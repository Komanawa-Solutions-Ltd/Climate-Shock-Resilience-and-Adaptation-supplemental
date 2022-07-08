"""
created matt_dumont 
on: 8/07/22
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

default_mode_sites = (
    ('dryland', 'oxford'),
    ('irrigated', 'eyrewell'),
    ('irrigated', 'oxford'),
    ('store400', 'eyrewell'),
    ('store400', 'oxford'),
    ('store600', 'eyrewell'),
    ('store600', 'oxford'),
    ('store800', 'eyrewell'),
    ('store800', 'oxford'),

)

data_dir = Path(__file__).parent


def extract_normalised_data(outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)
    for mode, site in default_mode_sites:
        sm = f'{site}-{mode}'
        data = pd.read_csv(data_dir.joinpath(f'{sm}-PGR-daily_total-singe_events.csv'), index_col=0)

        # normalise data to base year
        cols = np.arange(0, 24).astype(str)
        for m in data.month.unique():
            idx = data.month == m
            print(m)
            if m in [5, 6, 7, 8]:
                base_key = f'm{m:02d}-A-A-0-{sm}'
            else:
                base_key = f'm{m:02d}-A-A-50-{sm}'

            data.loc[idx, cols] += -1 * data.loc[base_key, cols]
        data.to_csv(outdir)


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'expected python path and output path'
    extract_normalised_data(sys.argv[1])
