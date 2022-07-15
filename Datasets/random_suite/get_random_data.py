"""
this script contains 5 main functions to access the datasets:

* get_1yr_data:
    This function get the random suite of 1yr storylines with BASGRA results for
* get storyline:
* create_nyr_suite:
    this resamples the random suite of BASGRA results in accordance to their probability to make 1 or more year
    synthetic timeseries.  These data can be large and therefore need to be re-calculated by the end user
* get_nyr_data
    This function is made to get the nyr datasets after the user runs them
* get_nyr_idxs
    this function is made to get the index values for the nyr suites (e.g. which storylines made up the nyr suite)

created matt_dumont 
on: 15/07/22
"""
from pathlib import Path
import pandas as pd
import numpy as np
import psutil
import gc
from zipfile import ZipFile


def get_1yr_data(bad_irr=True, good_irr=True, farm_mods=False):
    """
    get the raw random 1 year data.  Note  that the distribution of these data does not match the distribution
    of 1 year impacts as some storylines are more probable than others.
    :param bad_irr: bool if True return the data from the worse than median irrigation restriction suite
    :param good_irr: bool if True return the data from the better than median irrigation restriction suite
    :param farm_mods: bool if True add farm consultant modifications, if False then use raw BASGRA output.
    :return:
    """
    base_data_path = Path(__file__).parent.joinpath('data')
    assert any([bad_irr, good_irr])
    good, bad = None, None
    if bad_irr:
        bad = pd.read_hdf(base_data_path.joinpath(f'IID_probs_pg_1y_bad_irr.hdf'), 'prob')

    if good_irr:
        good = pd.read_hdf(base_data_path.joinpath(f'IID_probs_pg_1y_good_irr.hdf'), 'prob')

    if farm_mods:
        data = pd.concat([good, bad])
        data = _add_farm_consultant_modifications(data, mode_site=default_mode_sites)
        return data
    else:
        return pd.concat([good, bad])


def get_storylines(ids, irr_types):
    """
    get the storylines for a given dataset
    :param ids: list of storyline ids (e.g., 'rsl-069998'])
    :param irr_types: a list of irr types one of:
                        * 'bad_irr': where irrigation restrictions range from 50th to 99th percentile
                        * 'good_irr': where irrigation restrictions range from 1st to 50th percentile

    :return:
    """
    outdata = []
    base_data_path = Path(__file__).parent.joinpath('data')
    assert len(ids) == len(irr_types), 'expected ids and irr_types to be the same length'
    for idd, itype in zip(ids, irr_types):
        if itype == 'good_irr':
            with ZipFile(base_data_path.joinpath('random_good_irr.zip')) as zf:
                with zf.open(f'random_good_irr/{idd}.csv') as f:
                    t = pd.read_csv(f)
            outdata.append(t)
        elif itype == 'bad_irr':
            with ZipFile(base_data_path.joinpath('random_bad_irr.zip')) as zf:
                with zf.open(f'random_bad_irr/{idd}.csv') as f:
                    t = pd.read_csv(f)
            outdata.append(t)
        else:
            raise ValueError(f'unexpected value in irr_types: {itype} expected ["good_irr" or "bad_irr"')
    return outdata


def create_nyr_suite(n, nyr, outdir, use_default_seed=True,
                     farm_mods=False, monthly_data=False):
    """
    this does keep the number consitant across sims as the same seed it used
    :param n: the number of iterations to run for the datasets we used n = int(2.5e8)
    :param outdir: directory to save the simulations to
    :param nyr: number of years long, options are: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50,]
    :param use_default_seed: bool if true then use the default seed to keep reproducibility
    :param farm_mods: bool if True add farm consultant modifications, if False then use raw BASGRA output.
    :param monthly_data: bool if True save monthly data
    :return:
    """
    print(f'nyear: {nyr}')
    assert isinstance(nyr, int)
    assert isinstance(n, int)
    if farm_mods:
        mod = 'farm_mods'
    else:
        mod = 'raw'
    if monthly_data:
        mthly = '_monthly'
    else:
        mthly = ''

    outdir = Path(outdir).joinpath(f'nyr_{nyr}_n{n}_{mod}{mthly}')
    outdir.mkdir(exist_ok=True, parents=True)

    out_paths = {
        f'{site}-{mode}': outdir.joinpath(f'IID_probs_pg_{nyr}y_{site}-{mode}.npy') for site, mode in
        default_mode_sites}
    out_idx_paths = {
        mode: outdir.joinpath(f'IID_stories_{nyr}y_{mode}.npy') for site, mode in
        default_mode_sites}

    data_1y = get_1yr_data(bad_irr=True, good_irr=True, farm_mods=farm_mods)
    data_1y.reset_index(inplace=True)
    data_1y.loc[:, 'ID'] = data_1y.loc[:, 'ID'] + '-' + data_1y.loc[:, 'irr_type']
    data_1y.set_index('ID', inplace=True)
    assert isinstance(data_1y, pd.DataFrame)
    data_1y = data_1y.dropna()

    default_seeds = {
        1: 654654,
        2: 471121,
        3: 44383,
        4: 80942,
        5: 464015,
        6: 246731,
        7: 229599,
        8: 182848,
        9: 310694,
        10: 367013,
        15: 458445,
        20: 448546,
        25: 788546,
        30: 547214,
        40: 457542,
        50: 544455,
    }
    if use_default_seed:
        seed = default_seeds[nyr]
    else:
        seed = np.random.randint(1, 500000)

    mem = psutil.virtual_memory().available - 3e9  # leave 3 gb spare
    total_mem_needed = np.zeros(1).nbytes * n * nyr * 4
    chunks = int(np.ceil(total_mem_needed / mem))
    print(f'running in {chunks} chunks')
    chunk_size = int(np.ceil(n / chunks))

    for mode, site in default_mode_sites:
        outpath = out_paths[f'{site}-{mode}']
        outpath_idx = out_idx_paths[mode]
        print('making dataframe')
        outdata = pd.DataFrame(index=range(n), columns=['log10_prob_dryland', 'log10_prob_irrigated',
                                                        f'{site}-{mode}_pg_yr{nyr}'
                                                        ], dtype=np.float32)
        print(outdata.dtypes)
        print('/n', mode, site)
        key = f'{site}-{mode}'
        np.random.seed(seed)

        if 'store' in mode:
            temp_p = 10 ** data_1y.loc[:, f'log10_prob_irrigated']
        else:
            temp_p = 10 ** data_1y.loc[:, f'log10_prob_{mode}']
        p = temp_p / temp_p.sum()
        idxs = np.random.choice(
            np.arange(len(data_1y), dtype=np.uint32),
            size=(n * nyr),
            p=p
        ).reshape((n, nyr))

        for c in range(chunks):
            print(f'chunk: {c}')
            start_idx = chunk_size * c
            end_idx = chunk_size * (c + 1)
            cs = chunk_size
            if c == chunks - 1:
                end_idx = n
                cs = end_idx - start_idx
            print('getting prob_dry')
            prob = data_1y[f'log10_prob_dryland'].values[idxs[start_idx:end_idx]]
            # note that I have changed the probability to be log10(probaility)
            outdata.loc[start_idx:end_idx - 1, f'log10_prob_dryland'] = prob.sum(axis=1).astype(np.float32)

            print('getting prob_irr')
            prob = data_1y[f'log10_prob_irrigated'].values[idxs[start_idx:end_idx]]
            # note that I have changed the probability to be log10(probaility)
            outdata.loc[start_idx:end_idx - 1, f'log10_prob_irrigated'] = prob.sum(axis=1).astype(np.float32)

            print('getting pg')
            if monthly_data:
                for m in range(1, 13):
                    pga = data_1y[f'{key}_pg_m{m:02d}'].values[idxs[start_idx:end_idx]].reshape(cs, nyr)
                    for y in range(nyr):
                        outdata.loc[start_idx:end_idx - 1, f'{key}_pg_yr{y}_m{m:02d}'] = pga[:, y]
            else:
                pga = data_1y[f'{key}_pg_yr1'].values[idxs[start_idx:end_idx]].reshape(cs, nyr)
                outdata.loc[start_idx:end_idx - 1, f'{key}_pg_yr{nyr}'] = pga.sum(axis=1).astype(np.float32)

        if not use_default_seed:
            print('recording indexes')
            for n in range(nyr):
                outdata.loc[:, f'scen_{n + 1}'] = idxs[:, n]

        print(f'saving {mode} - {site} to local drive for {nyr}y')
        np.save(outpath, outdata.values)
        with open(outpath.replace('.npy', '.csv'), 'w') as f:
            f.write(','.join(outdata.columns))
        if 'store' not in mode:
            np.save(outpath_idx, idxs)

        print(f'finished {mode} - {site}')
        gc.collect()
    pd.Series(data_1y.index).to_csv(outdir.joinpath('story_index.csv'))


def get_nyr_idxs(n, nyr, base_dir, mode, farm_mods=False, monthly_data=False):
    """
    get the indexes used for the nyr simulations only use after running create_nyr_suite
    :param n: number of simulations to run (must match parameters used in create_nyr_suite)
    :param nyr: number of years to simulate (must match parameters used in create_nyr_suite)
    :param base_dir: base directory for the data (must match parameters used in create_nyr_suite)
    :param mode: the dryland, irrigation, or storage mode (see default_mode_sites for options)
    :param farm_mods: bool if True add farm consultant modifications, if False then use raw BASGRA output.
    :param monthly_data: bool if True save monthly data
    :return:
    """
    assert isinstance(nyr, int)
    assert isinstance(n, int)
    if farm_mods:
        mod = 'farm_mods'
    else:
        mod = 'raw'
    if monthly_data:
        mthly = '_monthly'
    else:
        mthly = ''

    outdir = Path(base_dir).joinpath(f'nyr_{nyr}_n{n}_{mod}{mthly}')

    if 'store' in mode:
        mode = 'irrigated'

    idx_path = outdir.joinpath(f'story_index.csv')
    if not idx_path.exists():
        raise FileNotFoundError(
            'could not find index file, check that the basedir, monthly_data, n, and nyr are correct '
            f'and that you have run create_nyr_idxs.\n expected {idx_path} to exist')
    story_path = outdir.joinpath(f'IID_stories_{nyr}y_{mode}.npy')
    if not story_path.exists():
        raise FileNotFoundError(
            'could not find storyline file, check that the basedir, monthly_data, n, and nyr are correct '
            f'and that you have run create_nyr_idxs.\n expected {story_path} to exist')

    indexes = pd.read_csv(idx_path).loc[:, 'ID'].values
    stories = np.load(story_path)
    stories = indexes[stories]
    return stories


def get_nyr_suite(n, nyr, base_dir, site, mode, farm_mods=False, monthly_data=False):
    """
    get the indexes used for the nyr simulations only use after running create_nyr_suite
    :param n: number of simulations to run (must match parameters used in create_nyr_suite)
    :param nyr: number of years to simulate (must match parameters used in create_nyr_suite)
    :param base_dir: base directory for the data (must match parameters used in create_nyr_suite)
    :param site: the site one of: ['eyrewell', 'oxford']
    :param mode: the dryland, irrigation, or storage mode (see default_mode_sites for options)
    :param farm_mods: bool if True add farm consultant modifications, if False then use raw BASGRA output.
    :param monthly_data: bool if True save monthly data
    :return:
    """
    assert isinstance(nyr, int)
    assert isinstance(n, int)
    if farm_mods:
        mod = 'farm_mods'
    else:
        mod = 'raw'
    if monthly_data:
        mthly = '_monthly'
    else:
        mthly = ''

    outdir = Path(base_dir).joinpath(f'nyr_{nyr}_n{n}_{mod}{mthly}')

    outdir = Path(base_dir).joinpath(f'nyr_{nyr}_n{n}_{mod}')

    data_path = outdir.joinpath(f'IID_probs_pg_{nyr}y_{site}-{mode}.npy')
    if not data_path.exists():
        raise FileNotFoundError(
            'could not find storyline file, check that the basedir, monthly_data, n, and nyr are correct '
            f'and that you have run create_nyr_idxs.\n expected {data_path} to exist')
    header_path = outdir.joinpath(f'IID_probs_pg_{nyr}y_{site}-{mode}.csv')
    if not header_path.exists():
        raise FileNotFoundError(
            'could not find storyline file, check that the basedir, monthly_data, n, and nyr are correct '
            f'and that you have run create_nyr_idxs.\n expected {header_path} to exist')

    out = np.load(data_path)
    out = pd.DataFrame(out, columns=pd.read_csv(header_path).columns)
    return out


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

fixed_data = {
    ('dryland', 'oxford'): {
        6: 5 * 30,
        7: 5 * 31,
        8: 10 * 31
    },
    ('irrigated', 'eyrewell'): {
        6: 5 * 30,
        7: 10 * 31,
        8: 15 * 31
    },
    ('irrigated', 'oxford'): {
        6: 5 * 30,
        7: 5 * 31,
        8: 10 * 31
    },
}

deltas = {
    9: 1.4,
    10: 1.4,
    11: 1.,
    12: 0.8,
    1: 0.8,
    2: 0.8,
    3: 0.8,
    4: 1.,
    5: 1.,
}


def _add_farm_consultant_modifications(data, mode_site=default_mode_sites):
    """
    # note that fractions are of cumulative montly
    :param data:
    :return:
    """
    data = data.copy(deep=True)

    for mode, site in mode_site:
        use_mode = mode
        if 'store' in use_mode:
            use_mode = 'irrigated'
        for m in range(1, 13):
            if m in [6, 7, 8]:
                data.loc[:, f'{site}-{mode}_pg_m{m:02d}'] = fixed_data[(use_mode, site)][m]
            else:
                data.loc[:, f'{site}-{mode}_pg_m{m:02d}'] *= deltas[m]

        # 1 year
        data.loc[:, f'{site}-{mode}_pg_yr1'] = data.loc[:,
                                               [f'{site}-{mode}_pg_m{m:02d}' for m in range(1, 13)]].sum(axis=1)

    return data


if __name__ == '__main__':
    test = get_storylines(['rsl-069998', 'rsl-069998'],
                          ['good_irr', 'bad_irr'])
    pass
