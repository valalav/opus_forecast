use nalgebra::{DMatrix, DVector};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize)]
struct Row {
    date: String,
    y: f64,
    food: f64,
    nonfood: f64,
    services: f64,
    #[serde(default)]
    ki: Option<f64>,
    #[serde(default)]
    ruonia: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
struct Config {
    horizon: usize,
    #[serde(default)]
    train_start: Option<String>,
    #[serde(default)]
    cutoff: Option<String>,
    #[serde(default)]
    alpha: Option<f64>,
    #[serde(default)]
    epsilon: Option<f64>,
    #[serde(default)]
    max_iter: Option<usize>,
    #[serde(default)]
    tol: Option<f64>,
    #[serde(default)]
    use_macro: Option<bool>,
    #[serde(default)]
    ets_weights12: Option<Vec<f64>>,
    #[serde(default)]
    excluded_years: Option<Vec<i32>>,
    #[serde(default)]
    seasonal_excluded_years: Option<Vec<i32>>,
    #[serde(default)]
    min_train: Option<usize>,
    #[serde(default)]
    tariff_month_by_year: Option<BTreeMap<String, u32>>,
    #[serde(default)]
    future_components: Option<String>,
    #[serde(default)]
    future_rates: Option<String>,
    #[serde(default)]
    allow_nonconverged: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
struct Backtests {
    #[serde(default)]
    cutoffs: Vec<String>,
    #[serde(default)]
    targets: Vec<String>,
    horizons: Vec<usize>,
}
#[derive(Clone, Debug, Deserialize)]
struct Request {
    schema_version: u32,
    model: String,
    #[serde(default = "raw_frequency")]
    frequency: String,
    rows: Vec<Row>,
    config: Config,
    #[serde(default)]
    backtests: Option<Backtests>,
}
fn raw_frequency() -> String {
    "raw".into()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Month {
    y: i32,
    m: u32,
}
impl Month {
    fn parse(s: &str) -> Result<Self, String> {
        if s.len() != 10 || &s[7..] != "-01" {
            return Err(format!("date must be YYYY-MM-01: {s}"));
        }
        let y = s[0..4]
            .parse::<i32>()
            .map_err(|_| format!("invalid date: {s}"))?;
        let m = s[5..7]
            .parse::<u32>()
            .map_err(|_| format!("invalid date: {s}"))?;
        if !(1..=12).contains(&m) || s.as_bytes().get(4) != Some(&b'-') {
            return Err(format!("invalid date: {s}"));
        }
        Ok(Self { y, m })
    }
    fn next(self, n: usize) -> Self {
        let z = self.y * 12 + self.m as i32 - 1 + n as i32;
        Self {
            y: z.div_euclid(12),
            m: z.rem_euclid(12) as u32 + 1,
        }
    }
    fn previous(self, n: usize) -> Self {
        let z = self.y * 12 + self.m as i32 - 1 - n as i32;
        Self {
            y: z.div_euclid(12),
            m: z.rem_euclid(12) as u32 + 1,
        }
    }
    fn text(self) -> String {
        format!("{:04}-{:02}-01", self.y, self.m)
    }
}

#[derive(Clone)]
struct Prepared {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    names: Vec<String>,
    seasonal: [f64; 12],
    macro_medians: [f64; 4],
}
#[derive(Clone)]
struct Fit {
    coef: Vec<f64>,
    center: Vec<f64>,
    scale: Vec<f64>,
    fit_converged: bool,
    iterations: usize,
    objective: f64,
    huber_scale: f64,
    gradient_inf_norm: f64,
    features: Vec<String>,
    n: usize,
}

fn err(code: &str, msg: impl Into<String>) -> Value {
    json!({"schema_version":1,"ok":false,"error":{"code":code,"message":msg.into()}})
}
fn finite(v: f64, name: &str) -> Result<(), String> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(format!("{name} must be finite"))
    }
}
fn validate(req: &Request) -> Result<Vec<Month>, String> {
    if req.schema_version != 1 {
        return Err("schema_version must be 1".into());
    }
    if req.model != "ridge" && req.model != "huber" {
        return Err("model must be ridge or huber".into());
    }
    if req.frequency != "raw" && req.frequency != "sa" {
        return Err("frequency must be raw or sa".into());
    }
    let c = &req.config;
    if !(1..=24).contains(&c.horizon) {
        return Err("horizon must be between 1 and 24".into());
    }
    let alpha = c.alpha.unwrap_or(0.3);
    if !alpha.is_finite() || !(0.0..=1_000_000.0).contains(&alpha) {
        return Err("alpha must be finite and in [0, 1000000]".into());
    }
    let epsilon = c.epsilon.unwrap_or(1.35);
    if !epsilon.is_finite() || !(0.05..=10.0).contains(&epsilon) {
        return Err("epsilon must be finite and in [0.05, 10]".into());
    }
    let tol = c.tol.unwrap_or(1e-5);
    if !tol.is_finite() || !(1e-12..=1.0).contains(&tol) {
        return Err("tol must be finite and in [1e-12, 1]".into());
    }
    if !(1..=10_000).contains(&c.max_iter.unwrap_or(500)) {
        return Err("max_iter must be in [1, 10000]".into());
    }
    if let Some(w) = &c.ets_weights12 {
        if w.len() != 12 || w.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
            return Err("ets_weights12 must contain 12 finite weights in [0,1]".into());
        }
    }
    if c.future_components.as_deref().unwrap_or("hold_last") != "hold_last"
        || c.future_rates.as_deref().unwrap_or("hold_last") != "hold_last"
    {
        return Err("future_components and future_rates currently support only hold_last".into());
    }
    if c.min_train.unwrap_or(36) < 1 {
        return Err("min_train must be positive".into());
    }
    if c.min_train.unwrap_or(36) > 5_000 {
        return Err("min_train must be <= 5000".into());
    }
    if let Some(schedule) = &c.tariff_month_by_year {
        for (year, month) in schedule {
            if year.len() != 4 || year.parse::<i32>().is_err() || !(1..=12).contains(month) {
                return Err(
                    "tariff_month_by_year must map four-digit years to months 1..12".into(),
                );
            }
        }
    }
    let mut dates: Vec<Month> = Vec::new();
    for (i, r) in req.rows.iter().enumerate() {
        let d = Month::parse(&r.date)?;
        if i > 0 && d != dates[i - 1].next(1) {
            return Err(format!("monthly gap or unordered rows at {}", r.date));
        }
        for (v, n) in [
            (r.y, "y"),
            (r.food, "food"),
            (r.nonfood, "nonfood"),
            (r.services, "services"),
        ] {
            finite(v, n)?;
        }
        if let Some(v) = r.ki {
            finite(v, "ki")?
        }
        if let Some(v) = r.ruonia {
            finite(v, "ruonia")?
        }
        dates.push(d);
    }
    if req.rows.is_empty() {
        return Err("rows must not be empty".into());
    }
    if let Some(s) = &c.train_start {
        Month::parse(s)?;
    }
    if let Some(s) = &c.cutoff {
        Month::parse(s)?;
    }
    Ok(dates)
}

fn features_for(model: &str) -> Vec<String> {
    let mut v = if model == "ridge" {
        vec![
            "y_lag1",
            "y_lag2",
            "y_lag12",
            "y_ma3",
            "month_sin",
            "month_cos",
            "food_lag1",
            "nonfood_lag1",
            "services_lag1",
            "seasonal_norm",
            "deviation_lag1",
        ]
    } else {
        vec![
            "y_lag1",
            "y_lag2",
            "y_lag12",
            "y_lag3",
            "y_lag6",
            "y_ma3",
            "y_ma6",
            "d_y_lag1",
            "d_y_lag3",
            "y_vol3",
            "y_vol6",
            "month_sin",
            "month_cos",
            "quarter_sin",
            "quarter_cos",
            "is_jan",
            "is_dec",
            "is_tariff_month",
            "is_q1",
            "is_summer",
            "food_lag1",
            "nonfood_lag1",
            "services_lag1",
            "seasonal_norm",
            "deviation_lag1",
        ]
    };
    v.extend(["ruonia_diff_lag1", "spread_lag4", "ki_diff_lag6", "ki_vol"]);
    v.into_iter().map(str::to_owned).collect()
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}
fn quantile(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let i = p * (v.len() - 1) as f64;
    let lo = i.floor() as usize;
    let hi = i.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (i - lo as f64)
}
fn sd(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1).max(1) as f64).sqrt()
}
fn seasonals(rows: &[Row], dates: &[Month], skip: &BTreeSet<i32>) -> [f64; 12] {
    let mut out = [100.0; 12];
    for m in 1..=12 {
        let a: Vec<_> = rows
            .iter()
            .zip(dates)
            .filter(|(_, d)| d.m == m && !skip.contains(&d.y))
            .map(|(r, _)| r.y)
            .collect();
        if !a.is_empty() {
            out[(m - 1) as usize] = mean(&a)
        }
    }
    out
}
fn tariff_month(req: &Request, year: i32) -> u32 {
    req.config
        .tariff_month_by_year
        .as_ref()
        .and_then(|x| x.get(&year.to_string()))
        .copied()
        .unwrap_or(7)
}
fn feature_row(
    i: usize,
    rows: &[Row],
    dates: &[Month],
    seasonal: &[f64; 12],
    model: &str,
    tariff: u32,
) -> Option<Vec<f64>> {
    if i < 12 {
        return None;
    }
    let ys: Vec<f64> = rows.iter().map(|r| r.y).collect();
    let mut f = Vec::new();
    let ylag = |n: usize| ys[i - n];
    let month = dates[i].m;
    let quarter = ((month - 1) / 3 + 1) as f64;
    let avg = |n: usize| mean(&ys[i - n..i]);
    let vol = |n: usize| sd(&ys[i - n..i]);
    let s = |m: u32| seasonal[(m - 1) as usize];
    if model == "ridge" {
        f.extend([
            ylag(1),
            ylag(2),
            ylag(12),
            avg(3),
            (2.0 * std::f64::consts::PI * month as f64 / 12.0).sin(),
            (2.0 * std::f64::consts::PI * month as f64 / 12.0).cos(),
            rows[i - 1].food,
            rows[i - 1].nonfood,
            rows[i - 1].services,
            s(month),
            ylag(1) - s(dates[i - 1].m),
        ]);
    } else {
        f.extend([
            ylag(1),
            ylag(2),
            ylag(12),
            ylag(3),
            ylag(6),
            avg(3),
            avg(6),
            ylag(1) - ylag(2),
            ylag(1) - ylag(4),
            vol(3),
            vol(6),
            (2.0 * std::f64::consts::PI * month as f64 / 12.0).sin(),
            (2.0 * std::f64::consts::PI * month as f64 / 12.0).cos(),
            (2.0 * std::f64::consts::PI * quarter / 4.0).sin(),
            (2.0 * std::f64::consts::PI * quarter / 4.0).cos(),
            (month == 1) as u8 as f64,
            (month == 12) as u8 as f64,
            (month == tariff) as u8 as f64,
            (quarter == 1.0) as u8 as f64,
            (month >= 6 && month <= 8) as u8 as f64,
            rows[i - 1].food,
            rows[i - 1].nonfood,
            rows[i - 1].services,
            s(month),
            ylag(1) - s(dates[i - 1].m),
        ]);
    }
    if rows.iter().any(|r| r.ki.is_some() && r.ruonia.is_some()) {
        let ki: Vec<_> = rows.iter().map(|r| r.ki.unwrap_or(f64::NAN)).collect();
        let ru: Vec<_> = rows.iter().map(|r| r.ruonia.unwrap_or(f64::NAN)).collect();
        f.extend([
            ru[i - 1] - ru[i - 2],
            ki[i - 4] - ru[i - 4],
            ki[i - 6] - ki[i - 7],
            sd(&ki[i - 6..i]),
        ]);
    }
    Some(f)
}

fn prepare(req: &Request, rows: &[Row], dates: &[Month], model: &str) -> Result<Prepared, String> {
    let config = &req.config;
    let use_macro = config.use_macro.unwrap_or(true);
    let mut names = features_for(model);
    if !use_macro {
        names.truncate(names.len() - 4)
    }
    let excluded = config
        .excluded_years
        .clone()
        .unwrap_or(if model == "ridge" {
            vec![2010, 2022]
        } else {
            vec![]
        })
        .into_iter()
        .collect::<BTreeSet<_>>();
    let seasonal_skip = config
        .seasonal_excluded_years
        .clone()
        .unwrap_or(vec![2010, 2022])
        .into_iter()
        .collect::<BTreeSet<_>>();
    let seasonal = seasonals(rows, dates, &seasonal_skip);
    let train_start = config
        .train_start
        .as_deref()
        .map(Month::parse)
        .transpose()?;
    let cutoff = config.cutoff.as_deref().map(Month::parse).transpose()?;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut macro_columns: [Vec<f64>; 4] = Default::default();
    for i in 12..rows.len() {
        if excluded.contains(&dates[i].y)
            || train_start.is_some_and(|v| dates[i] < v)
            || cutoff.is_some_and(|v| dates[i] > v)
        {
            continue;
        }
        let Some(mut f) = feature_row(
            i,
            rows,
            dates,
            &seasonal,
            model,
            tariff_month(req, dates[i].y),
        ) else {
            continue;
        };
        if f[..if use_macro { f.len() - 4 } else { f.len() }]
            .iter()
            .any(|v| !v.is_finite())
        {
            continue;
        }
        if !use_macro {
            f.truncate(names.len())
        } else {
            for j in 0..4 {
                let v = f[names.len() - 4 + j];
                if v.is_finite() {
                    macro_columns[j].push(v)
                }
            }
        }
        x.push(f);
        y.push(rows[i].y);
    }
    if x.len() < config.min_train.unwrap_or(36) {
        return Err(format!(
            "insufficient training rows: {} < {}",
            x.len(),
            config.min_train.unwrap_or(36)
        ));
    }
    let macro_medians = std::array::from_fn(|j| {
        if macro_columns[j].is_empty() {
            0.0
        } else {
            median(macro_columns[j].clone())
        }
    });
    if use_macro {
        for row in &mut x {
            for j in 0..4 {
                let k = row.len() - 4 + j;
                if !row[k].is_finite() {
                    row[k] = macro_medians[j]
                }
            }
        }
    }
    Ok(Prepared {
        x,
        y,
        names,
        seasonal,
        macro_medians,
    })
}

fn robust_scale(p: &Prepared) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let d = p.x[0].len();
    let mut center = vec![0.; d];
    let mut scale = vec![1.; d];
    for j in 0..d {
        let v: Vec<_> = p.x.iter().map(|r| r[j]).collect();
        center[j] = median(v.clone());
        let q = quantile(v.clone(), 0.75) - quantile(v, 0.25);
        if q > 1e-12 {
            scale[j] = q
        }
    }
    let xs =
        p.x.iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, v)| (v - center[j]) / scale[j])
                    .collect()
            })
            .collect();
    (xs, center, scale)
}
fn ridge_fit(p: &Prepared, alpha: f64) -> Fit {
    let (xs, center, scale) = robust_scale(p);
    let n = xs.len();
    let d = xs[0].len();
    let mut a = DMatrix::zeros(d + 1, d + 1);
    let mut b = DVector::zeros(d + 1);
    for (r, y) in xs.iter().zip(&p.y) {
        let mut z = vec![1.0];
        z.extend(r);
        for i in 0..d + 1 {
            b[i] += z[i] * y;
            for j in 0..d + 1 {
                a[(i, j)] += z[i] * z[j]
            }
        }
    }
    for j in 1..d + 1 {
        a[(j, j)] += alpha
    }
    let coef = a
        .clone()
        .lu()
        .solve(&b)
        .unwrap_or_else(|| DVector::zeros(d + 1));
    let obj = (0..n)
        .map(|i| {
            let pr: f64 = coef[0] + (0..d).map(|j| coef[j + 1] * xs[i][j]).sum::<f64>();
            (pr - p.y[i]).powi(2)
        })
        .sum::<f64>()
        + alpha * (1..coef.len()).map(|i| coef[i] * coef[i]).sum::<f64>();
    Fit {
        coef: coef.iter().copied().collect(),
        center,
        scale,
        fit_converged: true,
        iterations: 1,
        objective: obj,
        huber_scale: 0.,
        gradient_inf_norm: 0.,
        features: p.names.clone(),
        n,
    }
}

fn huber_obj(theta: &[f64], x: &[Vec<f64>], y: &[f64], eps: f64, alpha: f64) -> (f64, Vec<f64>) {
    let d = x[0].len();
    let sigma = theta[d + 1].exp().clamp(1e-8, 1e8);
    let mut obj = y.len() as f64 * sigma;
    let mut grad = vec![0.; d + 2];
    let mut gs = y.len() as f64 * sigma;
    for (r, yy) in x.iter().zip(y) {
        let pred = theta[0] + (0..d).map(|j| theta[j + 1] * r[j]).sum::<f64>();
        let e = yy - pred;
        let ae = e.abs();
        let (loss, dl_dpred, dl_dlogsigma) = if ae <= eps * sigma {
            (e * e / sigma, -2.0 * e / sigma, -(e * e) / sigma)
        } else {
            (
                2.0 * eps * ae - eps * eps * sigma,
                -2.0 * eps * e.signum(),
                -eps * eps * sigma,
            )
        };
        obj += loss;
        grad[0] += dl_dpred;
        for j in 0..d {
            grad[j + 1] += dl_dpred * r[j]
        }
        gs += dl_dlogsigma;
    }
    for j in 1..d + 1 {
        obj += alpha * theta[j] * theta[j];
        grad[j] += 2.0 * alpha * theta[j]
    }
    grad[d + 1] = gs;
    (obj, grad)
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn huber_fit(p: &Prepared, alpha: f64, eps: f64, max_iter: usize, tol: f64) -> Fit {
    let (xs, center, scale) = robust_scale(p);
    let d = xs[0].len();
    let mut t = vec![mean(&p.y)];
    t.resize(d + 1, 0.);
    t.push(sd(&p.y).max(1e-4).ln());
    let (mut f, mut g) = huber_obj(&t, &xs, &p.y, eps, alpha);
    let mut h = DMatrix::<f64>::identity(d + 2, d + 2);
    let mut converged = false;
    let mut it = 0;
    for k in 0..max_iter {
        it = k + 1;
        if g.iter().map(|v| v.abs()).fold(0.0, f64::max) < tol {
            converged = true;
            break;
        }
        let gv = DVector::from_vec(g.clone());
        let direction = -(h.clone() * gv);
        let mut dir: Vec<f64> = direction.iter().copied().collect();
        if dot(&dir, &g) >= 0. {
            h = DMatrix::identity(d + 2, d + 2);
            dir = g.iter().map(|v| -v).collect()
        }
        let mut step = 1.;
        let mut candidate = t.clone();
        let mut nf = f;
        let mut ng = g.clone();
        let mut found = false;
        for _ in 0..40 {
            candidate = t.iter().zip(&dir).map(|(a, b)| a + step * b).collect();
            let pair = huber_obj(&candidate, &xs, &p.y, eps, alpha);
            nf = pair.0;
            ng = pair.1;
            if nf <= f + 1e-4 * step * dot(&g, &dir) {
                found = true;
                break;
            }
            step *= 0.5
        }
        if !found {
            break;
        }
        let sv: Vec<_> = candidate.iter().zip(&t).map(|(a, b)| a - b).collect();
        let yv: Vec<_> = ng.iter().zip(&g).map(|(a, b)| a - b).collect();
        let ys = dot(&yv, &sv);
        if ys > 1e-12 {
            let rho = 1.0 / ys;
            let s = DVector::from_vec(sv);
            let yy = DVector::from_vec(yv);
            let ident = DMatrix::<f64>::identity(d + 2, d + 2);
            h = (ident.clone() - rho * &s * yy.transpose())
                * h
                * (ident - rho * &yy * s.transpose())
                + rho * &s * s.transpose()
        }
        t = candidate;
        f = nf;
        g = ng;
    }
    let gradient_inf_norm = g.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let sigma = t[d + 1].exp().clamp(1e-8, 1e8);
    Fit {
        coef: t[..d + 1].to_vec(),
        center,
        scale,
        fit_converged: converged,
        iterations: it,
        objective: f,
        huber_scale: sigma,
        gradient_inf_norm,
        features: p.names.clone(),
        n: xs.len(),
    }
}
fn pred(fit: &Fit, raw: &[f64]) -> f64 {
    fit.coef[0]
        + raw
            .iter()
            .enumerate()
            .map(|(j, v)| fit.coef[j + 1] * (v - fit.center[j]) / fit.scale[j])
            .sum::<f64>()
}

fn fit_model(req: &Request, history: &[Row], dates: &[Month]) -> Result<(Fit, Prepared), String> {
    let model = &req.model;
    let use_macro = req.config.use_macro.unwrap_or(true);
    if use_macro {
        if history.len() < 12
            || history[history.len() - 12..].iter().any(|r| {
                r.ki.is_none_or(|v| !v.is_finite()) || r.ruonia.is_none_or(|v| !v.is_finite())
            })
        {
            return Err("macro freshness requirement failed: latest 12 training months must have finite ki and ruonia; set use_macro=false to run without macro features".into());
        }
    }
    let p = prepare(req, history, dates, model)?;
    let alpha = req.config.alpha.unwrap_or(0.3);
    let fit = if model == "ridge" {
        ridge_fit(&p, alpha)
    } else {
        huber_fit(
            &p,
            alpha,
            req.config.epsilon.unwrap_or(1.35),
            req.config.max_iter.unwrap_or(500),
            req.config.tol.unwrap_or(1e-5),
        )
    };
    if !fit.fit_converged && !req.config.allow_nonconverged.unwrap_or(false) {
        return Err(format!(
            "Huber optimizer status=nonconverged after {} iterations; objective={:.12}; scale={:.12}; set allow_nonconverged=true only to inspect this fit without treating it as a valid forecast",
            fit.iterations, fit.objective, fit.huber_scale
        ));
    }
    Ok((fit, p))
}

fn forecast(
    req: &Request,
    input_rows: &[Row],
    input_dates: &[Month],
    origin: Month,
    horizon: usize,
) -> Result<(Vec<Value>, Fit), String> {
    let end = input_dates
        .iter()
        .position(|d| *d == origin)
        .ok_or_else(|| format!("cutoff {origin:?} is not present in rows"))?;
    let mut rows = input_rows[..=end].to_vec();
    let mut dates = input_dates[..=end].to_vec();
    let mut local = req.clone();
    local.config.cutoff = None;
    let (fit, p) = fit_model(&local, &rows, &dates)?;
    let mut steps = Vec::new();
    for h in 1..=horizon {
        let d = origin.next(h);
        let mut r = rows.last().unwrap().clone();
        r.date = d.text();
        r.food = rows.last().unwrap().food;
        r.nonfood = rows.last().unwrap().nonfood;
        r.services = rows.last().unwrap().services;
        r.ki = rows.last().unwrap().ki;
        r.ruonia = rows.last().unwrap().ruonia;
        let i = rows.len();
        dates.push(d);
        let mut f = feature_row(
            i,
            &rows,
            &dates,
            &p.seasonal,
            &req.model,
            tariff_month(req, d.y),
        )
        .ok_or_else(|| "unable to build forecast features".to_string())?;
        if req.config.use_macro.unwrap_or(true) {
            for j in 0..4 {
                let k = f.len() - 4 + j;
                if !f[k].is_finite() {
                    f[k] = p.macro_medians[j]
                }
            }
        } else {
            f.truncate(fit.features.len())
        }
        if f.len() != fit.features.len() {
            return Err("internal feature count mismatch".into());
        }
        let raw = pred(&fit, &f);
        let seasonal = p.seasonal[(d.m - 1) as usize];
        let weights = req.config.ets_weights12.as_ref();
        let w = weights.map(|v| v[(d.m - 1) as usize]).unwrap_or(
            [0.9, 0.0, 0.5, 0.3, 0.9, 0.5, 0.0, 0.5, 0.9, 0.9, 0.0, 0.0][(d.m - 1) as usize],
        );
        let level = (1. - w) * raw + w * seasonal;
        let all = level - 100.0;
        finite(all, "forecast")?;
        steps.push(json!({"date":d.text(),"all":all,"model_level":level,"ets_level":seasonal,"ets_weight":w}));
        r.y = level;
        rows.push(r);
    }
    Ok((steps, fit))
}

fn fit_json(f: &Fit) -> Value {
    json!({"converged":f.fit_converged,"iterations":f.iterations,"objective":f.objective,"scale":if f.huber_scale>0.0 {Value::from(f.huber_scale)} else {Value::Null},"gradient_inf_norm":f.gradient_inf_norm,"features":f.features,"n_train":f.n,"coefficients_intercept_then_features":f.coef,"scaler_center":f.center,"scaler_scale":f.scale})
}
fn run(req: Request) -> Result<Value, String> {
    let dates = validate(&req)?;
    if let Some(bt) = &req.backtests {
        if !bt.cutoffs.is_empty() && !bt.targets.is_empty() {
            return Err("backtests must specify cutoffs or targets, not both".into());
        }
        if bt.cutoffs.is_empty() && bt.targets.is_empty() {
            return Err("backtests requires at least one cutoff or target".into());
        }
        for d in bt.cutoffs.iter().chain(bt.targets.iter()) {
            Month::parse(d)?;
        }
        if bt.horizons.is_empty() || bt.horizons.iter().any(|h| ![1, 2, 12].contains(h)) {
            return Err("backtest horizons must be a nonempty selection from 1, 2, 12".into());
        }
    }
    let default_origin = *dates.last().unwrap();
    let origin = req
        .config
        .cutoff
        .as_deref()
        .map(Month::parse)
        .transpose()?
        .unwrap_or(default_origin);
    let (steps, fit) = forecast(&req, &req.rows, &dates, origin, req.config.horizon)?;
    let mut backtest_rows = Vec::new();
    let mut metrics = serde_json::Map::new();
    let mut warnings = Vec::new();
    if req
        .config
        .tariff_month_by_year
        .as_ref()
        .is_some_and(|m| !m.is_empty())
    {
        warnings.push("custom tariff_month_by_year affects main forecast only; rolling backtests use July default calendar".to_string())
    }
    if let Some(bt) = &req.backtests {
        for &h in &bt.horizons {
            let mut errs = Vec::new();
            let mut planned = 0;
            let pairs: Vec<(Month, Month)> = if !bt.targets.is_empty() {
                bt.targets
                    .iter()
                    .map(|t| {
                        let target = Month::parse(t)?;
                        Ok((target.previous(h), target))
                    })
                    .collect::<Result<_, String>>()?
            } else {
                bt.cutoffs
                    .iter()
                    .map(|c| {
                        let cutoff = Month::parse(c)?;
                        Ok((cutoff, cutoff.next(h)))
                    })
                    .collect::<Result<_, String>>()?
            };
            for (cm, td) in pairs {
                let c = cm.text();
                let target_text = td.text();
                planned += 1;
                let result = if dates.iter().any(|d| *d == cm) {
                    if let Some(ti) = dates.iter().position(|d| *d == td) {
                        let mut local = req.clone();
                        local.config.tariff_month_by_year = None;
                        match forecast(&local, &req.rows, &dates, cm, h) {
                            Ok((s, _)) => {
                                let actual = req.rows[ti].y - 100.0;
                                let predv = s[h - 1]["all"].as_f64().unwrap();
                                let e = predv - actual;
                                errs.push((e, predv, actual));
                                json!({"cutoff":c,"target_date":target_text,"horizon":h,"status":"ok","actual":actual,"forecast":predv,"error":e})
                            }
                            Err(e) => {
                                json!({"cutoff":c,"target_date":target_text,"horizon":h,"status":"unavailable","reason":e})
                            }
                        }
                    } else {
                        json!({"cutoff":c,"target_date":target_text,"horizon":h,"status":"unavailable","reason":"target date outside supplied rows"})
                    }
                } else {
                    json!({"cutoff":c,"target_date":target_text,"horizon":h,"status":"unavailable","reason":"cutoff date absent from supplied rows"})
                };
                backtest_rows.push(result)
            }
            let n = errs.len();
            let mae = if n > 0 {
                errs.iter().map(|(e, _, _)| e.abs()).sum::<f64>() / n as f64
            } else {
                0.0
            };
            let rmse = if n > 0 {
                (errs.iter().map(|(e, _, _)| e * e).sum::<f64>() / n as f64).sqrt()
            } else {
                0.0
            };
            let bias = if n > 0 {
                errs.iter().map(|(e, _, _)| *e).sum::<f64>() / n as f64
            } else {
                0.0
            };
            let hit = if n > 0 {
                errs.iter().filter(|(e, _, _)| e.abs() <= 0.5).count() as f64 / n as f64
            } else {
                0.0
            };
            metrics.insert(h.to_string(),json!({"n_planned":planned,"n_valid":n,"mae":if n>0{json!(mae)}else{Value::Null},"rmse":if n>0{json!(rmse)}else{Value::Null},"bias":if n>0{json!(bias)}else{Value::Null},"hit_within_0_5":if n>0{json!(hit)}else{Value::Null}}));
        }
    }
    Ok(
        json!({"schema_version":1,"ok":true,"model":req.model,"frequency":req.frequency,"forecast":{"origin":origin.text(),"steps":steps},"fit":fit_json(&fit),"backtest":{"rows":backtest_rows,"metrics":metrics},"warnings":warnings}),
    )
}

pub fn process(input: &str) -> String {
    let out = std::panic::catch_unwind(|| -> Value {
        match serde_json::from_str::<Request>(input) {
            Ok(req) => match run(req) {
                Ok(v) => v,
                Err(e) => err("invalid_or_fit_error", e),
            },
            Err(e) => err("invalid_request", e.to_string()),
        }
    })
    .unwrap_or_else(|_| err("internal_error", "unexpected kernel failure"));
    serde_json::to_string(&out).unwrap_or_else(|_|"{\"schema_version\":1,\"ok\":false,\"error\":{\"code\":\"serialization\",\"message\":\"failed to serialize response\"}}".into())
}

#[wasm_bindgen::prelude::wasm_bindgen]
pub fn process_json(request: &str) -> String {
    process(request)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> String {
        let mut rows = Vec::new();
        for i in 0..60 {
            let d = Month {
                y: 2018 + (i / 12) as i32,
                m: (i % 12 + 1) as u32,
            };
            let m = d.m as f64;
            let y = 100.0 + 0.1 * m + 0.002 * (i as f64);
            rows.push(json!({"date":d.text(),"y":y,"food":100.0+0.2*m,"nonfood":100.0+0.1*m,"services":100.0+0.05*m,"ki":10.0,"ruonia":8.0}));
        }
        json!({"schema_version":1,"model":"ridge","frequency":"raw","rows":rows,"config":{"horizon":2,"use_macro":false},"backtests":{"cutoffs":["2022-12-01","2023-12-01"],"horizons":[1,2,12]}}).to_string()
    }
    #[test]
    fn ridge_fixture_returns_finite_forecast() {
        let v: Value = serde_json::from_str(&process(&fixture())).unwrap();
        assert_eq!(v["ok"], true);
        assert!(v["forecast"]["steps"][0]["all"]
            .as_f64()
            .unwrap()
            .is_finite());
    }
    #[test]
    fn huber_nonconvergence_is_reported_as_error_by_default() {
        let mut request: Value = serde_json::from_str(&fixture()).unwrap();
        request["model"] = json!("huber");
        let response: Value = serde_json::from_str(&process(&request.to_string())).unwrap();
        assert_eq!(response["ok"], false);
        assert!(response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("optimizer status=nonconverged"));
    }
    #[test]
    fn macro_mode_reports_missing_latest_macro_window() {
        let mut request: Value = serde_json::from_str(&fixture()).unwrap();
        request["config"]["use_macro"] = json!(true);
        for row in request["rows"].as_array_mut().unwrap() {
            row["ki"] = Value::Null;
            row["ruonia"] = Value::Null;
        }
        let response: Value = serde_json::from_str(&process(&request.to_string())).unwrap();
        assert_eq!(response["ok"], false);
        assert!(response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("macro freshness requirement failed"));
    }
    #[test]
    fn ridge_linear_fixture_matches_closed_form() {
        let p = Prepared {
            x: vec![vec![0.0], vec![1.0], vec![2.0]],
            y: vec![2.0, 5.0, 8.0],
            names: vec!["x".into()],
            seasonal: [100.0; 12],
            macro_medians: [0.0; 4],
        };
        let f = ridge_fit(&p, 0.0);
        assert!((pred(&f, &[1.0]) - 5.0).abs() < 1e-10);
        assert!((f.coef[0] - 5.0).abs() < 1e-10);
        assert!((f.coef[1] - 3.0).abs() < 1e-10);
    }
    #[test]
    fn huber_concomitant_gradient_matches_finite_difference() {
        let x = vec![vec![-1.0], vec![0.5], vec![2.0], vec![4.0]];
        let y = vec![1.0, 2.2, 5.1, 25.0];
        let theta = vec![2.0, 1.2, 0.4_f64.ln()];
        let (_, grad) = huber_obj(&theta, &x, &y, 1.35, 0.3);
        for j in 0..theta.len() {
            let mut plus = theta.clone();
            let mut minus = theta.clone();
            plus[j] += 1e-6;
            minus[j] -= 1e-6;
            let numeric = (huber_obj(&plus, &x, &y, 1.35, 0.3).0
                - huber_obj(&minus, &x, &y, 1.35, 0.3).0)
                / 2e-6;
            assert!(
                (grad[j] - numeric).abs() < 1e-4,
                "gradient {j}: {} != {numeric}",
                grad[j]
            );
        }
    }
    #[test]
    fn future_rows_do_not_change_cutoff_forecast() {
        let mut v: Value = serde_json::from_str(&fixture()).unwrap();
        v["config"]["cutoff"] = json!("2022-12-01");
        let a: Value = serde_json::from_str(&process(&v.to_string())).unwrap();
        let rows = v["rows"].as_array_mut().unwrap();
        for r in rows.iter_mut().skip(60) {
            r["y"] = json!(900.0)
        }
        let b: Value = serde_json::from_str(&process(&v.to_string())).unwrap();
        assert_eq!(
            a["forecast"]["steps"][0]["all"],
            b["forecast"]["steps"][0]["all"]
        );
    }
    #[test]
    fn backtest_targets_derive_each_horizon_cutoff() {
        let mut request: Value = serde_json::from_str(&fixture()).unwrap();
        request["backtests"] = json!({"targets":["2022-12-01"],"horizons":[1,2,12]});
        let response: Value = serde_json::from_str(&process(&request.to_string())).unwrap();
        assert_eq!(response["ok"], true);
        let rows = response["backtest"]["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        for row in rows {
            assert_eq!(row["target_date"], "2022-12-01");
            assert_eq!(row["status"], "ok");
            let h = row["horizon"].as_u64().unwrap() as usize;
            assert_eq!(
                row["cutoff"],
                Month::parse("2022-12-01").unwrap().previous(h).text()
            );
        }
    }
    #[test]
    fn backtest_rejects_cutoffs_and_targets_together() {
        let mut request: Value = serde_json::from_str(&fixture()).unwrap();
        request["backtests"]["targets"] = json!(["2022-12-01"]);
        let response: Value = serde_json::from_str(&process(&request.to_string())).unwrap();
        assert_eq!(response["ok"], false);
        assert!(response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not both"));
    }
}
