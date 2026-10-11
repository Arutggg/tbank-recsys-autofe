"""
Обёртка градиентного бустинга и протокол подбора гиперпараметров.

Два решения, от которых зависит честность всего сравнения.

1. ТЮНИНГ КАЖДОМУ МЕТОДУ С ОДИНАКОВЫМ БЮДЖЕТОМ. Задание требует обыграть
   «хорошо затюненный бустинг», поэтому сравнивать затюненный компилятор с
   дефолтным ручным набором нельзя. Каждому набору признаков выдаётся
   одинаковое число испытаний случайного поиска на одной и той же val-выборке.

2. ОТБОР ПО LOG-LOSS, А НЕ ПО AUC. В эксперименте 2 замерено, что AUC
   насыщается и не различает признаки с разным содержанием информации
   (0.6д и 12д дают одинаковый AUC при 57% и 38% нулей). Log-loss —
   строгое правило скоринга, оно чувствительно к калибровке и к значениям,
   а не только к порядку. Ранжирующие метрики считаются и отчитываются, но
   решение о модели принимается по log-loss.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

import numpy as np

from .eval import all_metrics, logloss


@dataclass
class GBDTConfig:
    """Гиперпараметры LightGBM. Значения по умолчанию — разумная середина."""

    n_estimators: int = 3000
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 50
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    max_depth: int = -1
    early_stopping_rounds: int = 100
    seed: int = 0
    #: Число потоков LightGBM. По умолчанию половина ядер, а не все.
    #:
    #: Причина измерена на этой машине: MacBook Air M2 пассивно охлаждаем,
    #: и при длительной нагрузке на все восемь ядер macOS уходит в
    #: троттлинг — kernel_task начинает занимать до 180% CPU, принудительно
    #: охлаждая систему, и полезная пропускная способность падает примерно
    #: вчетверо. Половина ядер держит машину в тепловом бюджете и на
    #: длинных прогонах даёт БОЛЬШЕ суммарной работы, чем попытка занять всё.
    n_jobs: int = max(1, (os.cpu_count() or 8) // 2)

    def to_lgb(self) -> dict:
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "max_depth": self.max_depth,
            "verbosity": -1,
            "seed": self.seed,
            "num_threads": self.n_jobs,
            "deterministic": True,
            "force_row_wise": True,
        }


#: Пространство случайного поиска. Границы подобраны так, чтобы покрыть
#: и «мелкие частые деревья», и «крупные редкие» — на разреженных
#: счётчиках оптимум смещается между этими режимами.
SEARCH_SPACE = {
    "learning_rate": (0.02, 0.15),
    "num_leaves": (15, 255),
    "min_child_samples": (10, 300),
    "feature_fraction": (0.4, 1.0),
    "bagging_fraction": (0.6, 1.0),
    "lambda_l2": (0.1, 20.0),
}


def sample_config(rng: np.random.Generator, base: GBDTConfig) -> GBDTConfig:
    cfg = GBDTConfig(**asdict(base))
    cfg.learning_rate = float(np.exp(rng.uniform(*np.log(SEARCH_SPACE["learning_rate"]))))
    cfg.num_leaves = int(np.exp(rng.uniform(*np.log(SEARCH_SPACE["num_leaves"]))))
    cfg.min_child_samples = int(np.exp(rng.uniform(*np.log(SEARCH_SPACE["min_child_samples"]))))
    cfg.feature_fraction = float(rng.uniform(*SEARCH_SPACE["feature_fraction"]))
    cfg.bagging_fraction = float(rng.uniform(*SEARCH_SPACE["bagging_fraction"]))
    cfg.lambda_l2 = float(np.exp(rng.uniform(*np.log(SEARCH_SPACE["lambda_l2"]))))
    return cfg


_SAFE_TRANS = str.maketrans({c: "_" for c in '[]{}":,|/*()\\ '})


def sanitize_names(names) -> list[str] | None:
    """LightGBM не принимает служебные символы JSON в именах признаков.

    Читаемые имена вида dcount[userxcat|purchase|7d] нужны в отчёте, поэтому
    здесь делается отдельное безопасное представление. Порядок сохраняется,
    так что важность признаков всегда можно сопоставить с исходным именем
    по индексу.
    """
    if names is None:
        return None
    safe, seen = [], {}
    for n in names:
        base = n.translate(_SAFE_TRANS)
        if base in seen:
            seen[base] += 1
            base = f"{base}__{seen[base]}"
        else:
            seen[base] = 0
        safe.append(base)
    return safe


def fit_gbdt(X_tr, y_tr, X_val, y_val, cfg: GBDTConfig, feature_names=None):
    """Обучить LightGBM с ранней остановкой по val log-loss."""
    import lightgbm as lgb

    dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=sanitize_names(feature_names),
                      free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr, free_raw_data=False)
    booster = lgb.train(
        cfg.to_lgb(), dtr, num_boost_round=cfg.n_estimators,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                   lgb.log_evaluation(0)],
    )
    return booster


def tune_gbdt(
    X_tr, y_tr, X_val, y_val, n_trials: int = 12, seed: int = 0,
    base: GBDTConfig | None = None, feature_names=None, verbose: bool = False,
) -> tuple[GBDTConfig, object, float]:
    """Случайный поиск с фиксированным бюджетом. Отбор по val log-loss.

    Первое испытание — всегда дефолтная конфигурация, чтобы бюджет поиска
    не мог сделать результат ХУЖЕ разумной середины.
    """
    rng = np.random.default_rng(seed)
    base = base or GBDTConfig(seed=seed)
    best = (None, None, np.inf)
    for trial in range(n_trials):
        cfg = base if trial == 0 else sample_config(rng, base)
        booster = fit_gbdt(X_tr, y_tr, X_val, y_val, cfg, feature_names)
        p = booster.predict(X_val, num_iteration=booster.best_iteration)
        ll = logloss(y_val, p)
        if verbose:
            print(f"    испытание {trial + 1}/{n_trials}: val logloss={ll:.5f} "
                  f"(lr={cfg.learning_rate:.3f}, leaves={cfg.num_leaves}, "
                  f"iters={booster.best_iteration})", flush=True)
        if ll < best[2]:
            best = (cfg, booster, ll)
    return best


@dataclass
class MethodResult:
    """Результат одного метода: метрики, стоимость, размер набора признаков."""

    name: str
    n_features: int
    test_metrics: dict = field(default_factory=dict)
    val_logloss: float = float("nan")
    feature_seconds: float = float("nan")
    train_seconds: float = float("nan")
    best_iteration: int = 0
    config: dict = field(default_factory=dict)
    human_hours: float = float("nan")   # оценка ручного труда на метод
    #: предсказания на test. Нужны, чтобы посчитать доверительные интервалы
    #: бутстрэпом по группам: различия между методами здесь порядка
    #: 0.001-0.005, и без интервала такие числа ничего не значат.
    test_pred: object = None

    def row(self) -> dict:
        d = {"метод": self.name, "признаков": self.n_features}
        d.update({k: v for k, v in self.test_metrics.items()})
        d["val_logloss"] = self.val_logloss
        d["фичи_сек"] = self.feature_seconds
        d["обучение_сек"] = self.train_seconds
        return d


def evaluate_feature_set(
    name: str,
    X_tr, y_tr, X_val, y_val, X_test, y_test, group_test,
    n_trials: int = 12, seed: int = 0, feature_names=None,
    feature_seconds: float = float("nan"), human_hours: float = float("nan"),
    verbose: bool = True,
) -> MethodResult:
    """Затюнить бустинг на наборе признаков и замерить метрики на test."""
    import time as _time

    t0 = _time.perf_counter()
    cfg, booster, val_ll = tune_gbdt(
        X_tr, y_tr, X_val, y_val, n_trials=n_trials, seed=seed,
        feature_names=feature_names, verbose=verbose)
    train_s = _time.perf_counter() - t0

    p = booster.predict(X_test, num_iteration=booster.best_iteration)
    metrics = all_metrics(y_test, p, group_test, proba=p)

    return MethodResult(
        name=name, n_features=X_tr.shape[1], test_metrics=metrics,
        val_logloss=val_ll, feature_seconds=feature_seconds,
        train_seconds=train_s, best_iteration=int(booster.best_iteration),
        config={k: v for k, v in asdict(cfg).items()
                if k in SEARCH_SPACE or k == "n_estimators"},
        human_hours=human_hours, test_pred=p,
    )


def select_by_importance(
    X_tr, y_tr, X_val, y_val, cfg: GBDTConfig,
    candidate_k=(50, 100, 200, 400, 800), feature_names=None, verbose=False,
) -> tuple[np.ndarray, int, float]:
    """Отбор признаков по важности. ВСЕ решения принимаются на val.

    Зачем это нужно. В первом прогоне бенчмарка полный перебор (1944
    признака) ПРОИГРАЛ ручному набору (98 признаков) по всем метрикам.
    Причина статистическая, а не логическая: при 120 тысячах обучающих
    строк отношение строк к признакам падает с 1200:1 до 60:1, и лишние
    признаки добавляют дисперсию быстрее, чем сигнал. Перебор без отбора
    действительно вредит — это и есть ответ.

    Процедура:
      1. обучить модель на всех признаках, ранняя остановка по val;
      2. взять важность по gain;
      3. перебрать несколько размеров K, для каждого переобучить и
         замерить val log-loss;
      4. вернуть маску лучшего K.

    Test в этой процедуре не участвует вообще.
    """
    booster = fit_gbdt(X_tr, y_tr, X_val, y_val, cfg, feature_names)
    imp = booster.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1]

    best = (np.ones(X_tr.shape[1], dtype=bool), X_tr.shape[1], np.inf)
    p_all = booster.predict(X_val, num_iteration=booster.best_iteration)
    best = (best[0], best[1], logloss(y_val, p_all))
    if verbose:
        print(f"      все {X_tr.shape[1]} признаков: val logloss={best[2]:.5f}")

    for k in candidate_k:
        if k >= X_tr.shape[1]:
            continue
        mask = np.zeros(X_tr.shape[1], dtype=bool)
        mask[order[:k]] = True
        names_k = [feature_names[i] for i in np.flatnonzero(mask)] if feature_names else None
        b = fit_gbdt(X_tr[:, mask], y_tr, X_val[:, mask], y_val, cfg, names_k)
        ll = logloss(y_val, b.predict(X_val[:, mask], num_iteration=b.best_iteration))
        if verbose:
            print(f"      top-{k}: val logloss={ll:.5f}")
        if ll < best[2]:
            best = (mask, k, ll)
    return best


def evaluate_with_selection(
    name: str,
    X_tr, y_tr, X_val, y_val, X_test, y_test, group_test,
    n_trials: int = 12, seed: int = 0, feature_names=None,
    feature_seconds: float = float("nan"), human_hours: float = float("nan"),
    candidate_k=(50, 100, 200, 400, 800), verbose: bool = True,
) -> MethodResult:
    """Полный пайплайн: перебор -> отбор по важности на val -> тюнинг -> test."""
    import time as _time

    t0 = _time.perf_counter()
    base = GBDTConfig(seed=seed)
    mask, k, _ = select_by_importance(
        X_tr, y_tr, X_val, y_val, base, candidate_k=candidate_k,
        feature_names=feature_names, verbose=verbose)
    names_sel = ([feature_names[i] for i in np.flatnonzero(mask)]
                 if feature_names else None)

    cfg, booster, val_ll = tune_gbdt(
        X_tr[:, mask], y_tr, X_val[:, mask], y_val, n_trials=n_trials,
        seed=seed, feature_names=names_sel, verbose=False)
    train_s = _time.perf_counter() - t0

    p = booster.predict(X_test[:, mask], num_iteration=booster.best_iteration)
    metrics = all_metrics(y_test, p, group_test, proba=p)
    return MethodResult(
        name=name, n_features=int(mask.sum()), test_metrics=metrics,
        val_logloss=val_ll, feature_seconds=feature_seconds,
        train_seconds=train_s, best_iteration=int(booster.best_iteration),
        config={"selected_k": int(mask.sum()), **{k_: v for k_, v in asdict(cfg).items()
                                                  if k_ in SEARCH_SPACE}},
        human_hours=human_hours, test_pred=p,
    )
