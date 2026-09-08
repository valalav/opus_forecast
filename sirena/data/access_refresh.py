"""Validate and publish one Access RAW vintage; retain independently newer SA."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import urllib.request
import urllib.parse
import zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd

PUBLIC_URL = 'https://disk.yandex.ru/d/i6czrcdNO0I7BQ'


def prepare_release(release, *, download=False):
    """Download the current public archive if requested, then export its tables."""
    release = Path(release)
    release.mkdir(parents=True, exist_ok=True)
    database = release / 'db_cpi_store.accdb'
    if download:
        api = 'https://cloud-api.yandex.net/v1/disk/public/resources'
        def request(endpoint, **params):
            query = urllib.parse.urlencode({'public_key': PUBLIC_URL, **params})
            with urllib.request.urlopen(api + endpoint + '?' + query, timeout=60) as response:
                return json.load(response)
        # Read the exact archive resource, avoiding root-listing pagination.
        info = request('', path='/db_cpi_store.zip')
        archive = release / 'db_cpi_store.zip'
        if not archive.exists() or sha256(archive) != info['sha256']:
            link = request('/download', path='/db_cpi_store.zip')['href']
            temporary = archive.with_suffix('.zip.part')
            with urllib.request.urlopen(link, timeout=60) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            if sha256(temporary) != info['sha256']:
                raise ValueError('Downloaded archive hash differs from public source')
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as source_zip:
            members = [n for n in source_zip.namelist() if Path(n).name.lower() == database.name]
            if len(members) != 1:
                raise ValueError('Expected exactly one Access database in release')
            temporary = database.with_suffix('.accdb.part')
            with source_zip.open(members[0]) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            temporary.replace(database)
        source = {'public_url':PUBLIC_URL, 'name':info['name'], 'modified':info['modified'],
                  'archive_sha256':info['sha256'], 'database_sha256':sha256(database),
                  'database_bytes':database.stat().st_size,
                  'z_max_day':subprocess.check_output(['mdb-export',str(database),'z_max_day'],text=True)}
        (release/'manifest.json').write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding='utf-8')
    source = json.loads((release/'manifest.json').read_text())
    if sha256(database) != source['database_sha256']:
        raise ValueError('Database differs from download manifest')
    extracts = release/'extracts'
    extracts.mkdir(exist_ok=True)
    hashes = {}
    for table in ['data_indices','data_seasonalised','weights','items_names','items_structure']:
        parts, regional = [], []
        with subprocess.Popen(['mdb-export',str(database),table],stdout=subprocess.PIPE) as proc:
            for chunk in pd.read_csv(proc.stdout,chunksize=200_000):
                if table == 'weights':
                    parts.append(chunk)
                if 'Region_code' in chunk:
                    chunk = chunk[chunk.Region_code.eq(7)]
                regional.append(chunk)
            if proc.wait() != 0:
                raise ValueError(f'mdb-export failed: {table}')
        frame = pd.concat(regional,ignore_index=True)
        frame.to_csv(extracts/(table+'.csv'),index=False)
        hashes[table] = sha256(extracts/(table+'.csv'))
        if table == 'weights':
            pd.concat(parts,ignore_index=True).to_csv(extracts/'weights_all.csv',index=False)
            hashes['weights_all'] = sha256(extracts/'weights_all.csv')
    (release/'extract_manifest.json').write_text(json.dumps({
        'database_sha256':source['database_sha256'], 'tables':hashes},indent=2),encoding='utf-8')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def refresh_access(release, data_dir, *, apply=False, proxy=None, proxy_manifest=None):
    release, data_dir = Path(release), Path(data_dir)
    source = json.loads((release / 'manifest.json').read_text())
    database = release / 'db_cpi_store.accdb'
    if sha256(database) != source['database_sha256']:
        raise ValueError('Access database hash differs from verified download')
    extraction = json.loads((release/'extract_manifest.json').read_text())
    if extraction['database_sha256'] != source['database_sha256']:
        raise ValueError('Extracts belong to a different Access database; rerun --extract')
    required = ['data_indices','data_seasonalised','weights','items_names','items_structure','weights_all']
    for name in required:
        if sha256(release/'extracts'/(name+'.csv')) != extraction['tables'].get(name):
            raise ValueError(f'Extract hash mismatch: {name}; rerun --extract')
    tables = {name: pd.read_csv(release / 'extracts' / (name + '.csv')) for name in
              ['data_indices', 'data_seasonalised', 'weights', 'items_names', 'items_structure']}
    indices, weights = tables['data_indices'].copy(), tables['weights'].copy()
    names, structure = tables['items_names'], tables['items_structure']
    for frame in [indices, weights, tables['data_seasonalised']]:
        if set(frame.Region_code) != {7}:
            raise ValueError('Expected only KBR region 7')
        frame['Date'] = pd.to_datetime(frame.Day, format='%m/%d/%y %H:%M:%S', errors='raise')
        if frame.duplicated(['Date', 'Item_code']).any():
            raise ValueError('Duplicate month/item in source')
    latest = indices.Date.max()
    observed = indices.pivot(index='Date', columns='Item_code', values='MoM')
    official = pd.read_csv(data_dir / 'inflation_data.csv', sep=';', decimal=',')
    official.index = pd.to_datetime(official.Date, dayfirst=True).dt.to_period('M').dt.to_timestamp()
    if official.index.max() != latest:
        raise ValueError('Monthly canonical source must first be updated to the same cutoff')
    discrepancies = {}
    for code, col in [(1, 'mom'), (2, 'Nonprod'), (3, 'Prod'), (4, 'Serv')]:
        reference = pd.to_numeric(official[col].astype(str).str.replace(',', '.'), errors='raise')
        actual = observed[code].reindex(reference.index)
        if actual.isna().any() or not np.isfinite(actual).all():
            raise ValueError(f'Missing aggregate observations: {code}')
        diff = float(np.max(np.abs(actual.to_numpy() - reference.to_numpy())))
        if diff > 1e-5:
            raise ValueError(f'Aggregate CPI mismatch {code}: {diff}')
        discrepancies[str(code)] = {'months': len(reference), 'max_abs_difference': diff}
    year_weights = weights[weights.Date.eq(pd.Timestamp(latest.year, 1, 1))]
    basket = structure[(structure.Item_type == 5) & (structure.Item_on == 1)].merge(
        year_weights[['Item_code', 'Weight_vertical']], on='Item_code', validate='one_to_one')
    basket = basket.merge(names, on='Item_code', validate='one_to_one')
    if not np.isclose(basket.Weight_vertical.sum(), 1., atol=1e-5):
        raise ValueError('Active basket weights do not sum to one')
    name_map = names.set_index('Item_code').Item_name
    sprav = pd.DataFrame({'Item_code': basket.Item_code, 'Товар': basket.Item_name,
        'Компонент': basket.Component.map(name_map), 'Субкомпонент': basket.Subcomponent.map(name_map),
        'Weight': basket.Weight_vertical})
    current = observed.reindex(columns=basket.Item_code).loc[latest]
    missing = []
    for code in current[current.isna()].index:
        row = basket.set_index('Item_code').loc[code]
        last = observed[code].last_valid_index() if code in observed else None
        missing.append({'item_code': int(code), 'name': row.Item_name,
                        'weight': float(row.Weight_vertical), 'primary': int(row.Primary),
                        'last_observation': str(last.date()) if last is not None else None})
    outputs = {}
    def csv(name, frame, **kwargs):
        outputs[name] = frame.to_csv(index=False, lineterminator='\n', **kwargs).encode('utf-8-sig')
    csv('kbr_indices.csv', indices)
    csv('kbr_micro_full.csv', indices[indices.Item_code.isin(basket.Item_code)])
    csv('micro_sprav.csv', sprav, sep=';')
    all_weights = pd.read_csv(release / 'extracts' / 'weights_all.csv')
    if all_weights.Region_code.nunique() < 2:
        raise ValueError('Extended weights reference must retain all regions')
    pd.testing.assert_frame_equal(all_weights[all_weights.Region_code.eq(7)].reset_index(drop=True),
                                  tables['weights'], check_dtype=False)
    csv('access_weights.csv', all_weights)
    csv('items_names.csv', names)
    csv('items_structure.csv', structure)
    for name, date_col, month_end in [('raw/subcomp.csv', 'Day', False), ('raw/sub_mom.csv', 'Date', True)]:
        columns = pd.read_csv(data_dir / name, sep=';', nrows=0).columns[1:]
        sub = (observed.reindex(columns=[int(c) for c in columns]) - 100).copy()
        sub.columns = columns
        if sub.iloc[-1].isna().any():
            raise ValueError(f'Subcomponent missing at cutoff: {name}')
        dates = sub.index + pd.offsets.MonthEnd(0) if month_end else sub.index
        sub.insert(0, date_col, dates.strftime('%d.%m.%Y'))
        csv(name, sub, sep=';', decimal=',')
    long = indices.rename(columns={'Item_code':'item_code', 'Date':'date', 'MoM':'mom_index',
                                  'YoY':'yoy_index', 'Region_code':'region_code'})
    long = long[['date','item_code','mom_index','yoy_index','region_code']].merge(
        names.rename(columns={'Item_code':'item_code','Item_name':'item_name','Item_rosstat':'item_rosstat'}),
        on='item_code', validate='many_to_one')
    long['source_file'] = source['public_url'] + '/db_cpi_store.zip'
    long['region_name'] = 'Кабардино-Балкарская Республика'
    prefix = 'external/micro_cpi_region_export/'
    csv(prefix + 'region_cpi_long.csv', long)
    metadata = {'source_file': source['public_url'], 'database_sha256':source['database_sha256'],
        'extracted_at':datetime.now(timezone.utc).isoformat(), 'region_name':'Кабардино-Балкарская Республика',
        'region_code':'7','latest_month':str(latest.date()),'row_count':len(long),
        'item_count':int(long.item_code.nunique()),'representation':'raw', 'units':'index_previous_month_100',
        'extractor_version':'access_refresh_v1'}
    outputs[prefix + 'metadata.json'] = json.dumps(metadata, ensure_ascii=False, indent=2).encode()
    if proxy is not None:
        proxy, proxy_manifest = Path(proxy), Path(proxy_manifest)
        p = pd.read_csv(proxy, sep=';', decimal=',', index_col=0)
        p.index = pd.to_datetime(p.index, dayfirst=True)
        if p.index.max() != latest or not np.isfinite(p[['Torg','pp']].tail(12)).all().all():
            raise ValueError('Proxy does not reach CPI cutoff with valid observations')
        meta = json.loads(proxy_manifest.read_text(encoding='utf-8'))
        if meta.get('representation') != 'raw' or meta['measures']['pp'] != 'industrial_production':
            raise ValueError('Proxy representation/measure mismatch')
        outputs['raw/infostat.csv'] = proxy.read_bytes()
        outputs['raw/infostat_source_manifest.json'] = proxy_manifest.read_bytes()
    manifest = {'updated_at':datetime.now(timezone.utc).isoformat(), 'source':source,
        'raw_last_observation':str(latest.date()),
        'access_sa_last_observation':str(tables['data_seasonalised'].Date.max().date()),
        'sa_action':'RAW-only refresh; retain separately verified SA workbook vintage without splicing',
        'aggregate_crosscheck':discrepancies, 'basket_items':len(basket),
        'basket_observed_items':int(current.notna().sum()),
        'basket_observed_weight':float(1-sum(x['weight'] for x in missing)),
        'missing_items':missing, 'short_histories_lt36':int((observed.reindex(columns=basket.Item_code).count()<36).sum()),
        'extract_hashes':{n:sha256(release/'extracts'/(n+'.csv')) for n in [*tables, 'weights_all']},
        'outputs':{name:hashlib.sha256(value).hexdigest() for name,value in outputs.items()}}
    outputs['access_source_manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    if apply:
        backup = data_dir.parent/'archive/results'/datetime.now(timezone.utc).strftime('access_refresh_%Y%m%d/%H%M%S')
        backup.mkdir(parents=True, exist_ok=False)
        replaced = []
        try:
            for name, value in outputs.items():
                target = data_dir/name
                if target.exists():
                    old = backup/name
                    old.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(target, old)
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary = target.with_suffix(target.suffix+'.refresh-tmp')
                temporary.write_bytes(value)
                temporary.replace(target)
                replaced.append(name)
            target = data_dir/'db_cpi_store.accdb'
            if target.exists():
                shutil.copy2(target, backup/'db_cpi_store.accdb')
            replaced.append('db_cpi_store.accdb')
            temporary = target.with_suffix('.accdb.refresh-tmp')
            shutil.copy2(database, temporary)
            temporary.replace(target)
        except Exception:
            for name in replaced:
                old, target = backup/name, data_dir/name
                if old.exists():
                    shutil.copy2(old, target)
                elif target.exists():
                    target.unlink()
            raise
        manifest['backup'] = str(backup)
    return manifest
