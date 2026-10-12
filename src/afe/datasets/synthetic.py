"""
Управляемый генератор лога взаимодействий с ИЗВЕСТНЫМИ временными константами.

Зачем он нужен. Выше численно показано, что базис затуханий и жёсткие окна
не выражаются друг через друга устойчиво, поэтому вопрос «что лучше» не
решается теоретически — только экспериментом. Но на реальном датасете
«правильного» ответа не существует: мы видим только метрику, а истинное
ядро, которым устроено поведение людей, неизвестно.

Поэтому нужен полигон, где истина задана нами:

    tau_cat   — полупериод интереса пользователя к КАТЕГОРИИ
    tau_item  — полупериод склонности к ПОВТОРУ конкретного айтема
    tau_pop   — полупериод «трендовости» айтема

Генератор честно порождает события по этим константам. Тогда можно
спрос[ить] компилятор: на каком масштабе он найдёт сигнал? Если выученные
масштабы совпадут с заданными — метод валиден, а не просто «дал +0.01 AUC».
Такой проверки не даёт ни один публичный датасет.

Модель порождения
-----------------
Суперпозиция пуассоновских процессов по пользователям. Для события
пользователя u в момент t берётся набор кандидатов (из популярности +
недавняя история u), для каждого кандидата i считается

  logit(u,i,t) = b_pop_s * log1p(base_pop[i])            статическая популярность
               + b_pop_d * log1p(pop_decay[i])           тренд, полупериод tau_pop
               + b_cat   * log1p(aff_cat[u, cat[i]])      интерес к категории, tau_cat
               + b_brand * log1p(aff_brand[u, brand[i]])  интерес к бренду, tau_brand
               + b_item  * log1p(aff_item[u, i])          склонность к повтору, tau_item
               + gumbel noise

айтем выбирается softmax-сэмплированием, затем тип события (view / cart /
purchase) определяется той же силой сигнала: чем выше logit, тем вероятнее
эскалация до покупки. Это воспроизводит воронку из задания (0/1/2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numba
import numpy as np

from ..log import EventLog

if TYPE_CHECKING:                      # только для аннотаций, без циклического импорта
    from ..log import QuerySet

SEC_PER_DAY = 86400.0


@dataclass
class SyntheticConfig:
    """Параметры полигона. tau_* — та самая истина, которую мы прячем от метода."""

    n_users: int = 20_000
    n_items: int = 5_000
    n_cats: int = 40
    n_brands: int = 120
    n_price_buckets: int = 8
    n_events: int = 1_000_000
    span_days: float = 240.0

    #: ИСТИННЫЕ полупериоды (дни) — проверяем, восстановит ли их компилятор
    tau_cat_days: float = 45.0
    #: Полупериод интереса к БРЕНДУ. Бренд намеренно НЕ входит в набор
    #: ключей ручного baseline — это проверка главного заявления метода:
    #: найдёт ли компилятор полезную связь, которую человек не выписал.
    tau_brand_days: float = 20.0
    tau_item_days: float = 4.0
    tau_pop_days: float = 10.0

    #: веса слагаемых в logit. Держатся малыми намеренно: слагаемые должны
    #: быть сопоставимы с gumbel-шумом (масштаб ~1), иначе выбор становится
    #: детерминированным и лог вырождается в несколько пар user-item.
    b_pop_static: float = 0.55
    b_pop_decay: float = 0.15
    b_cat: float = 1.10
    #: Вес интереса к БРЕНДУ. Ненулевой — значит в данных есть сигнал,
    #: которого ручной набор (user, item, user x item, user x cat) не видит.
    #: Это и есть проверка главного заявления метода: найдёт ли компилятор
    #: полезную связь, которую человек не выписал.
    b_brand: float = 0.85
    b_item: float = 1.30

    #: насыщение повторов. Без него положительная обратная связь по
    #: w_rep уходит в разгон: чем чаще брал, тем выше вес, тем чаще берёт.
    rep_cap: float = 10.0
    #: пресыщение: штраф за айтем, тронутый совсем недавно (часы, не дни).
    satiation_halflife_days: float = 0.5
    b_satiation: float = 1.6

    n_candidates: int = 48
    n_recent: int = 24            # длина истории пользователя для повторов
    n_recent_candidates: int = 8  # сколько из них попадает в кандидаты
    zipf_a: float = 1.05          # хвост популярности айтемов
    #: Показатель, с которым популярность входит в ПУЛ КАНДИДАТОВ.
    #: При 1.0 популярность учитывается дважды — и в пуле, и в скоринге, —
    #: что даёт эффективно Ципф в квадрате и схлопывает каталог до десятков
    #: айтемов. Замерено: топ-10 забирали 40% массы сэмплирования.
    cand_pop_alpha: float = 0.4
    #: Обрезка стандартизованного члена популярности. Лог-популярность по
    #: Ципфу остаётся сильно скошенной даже после стандартизации
    #: (замерено: max z = 7.9 при медиане -0.36), и без обрезки один айтем
    #: получает +4.3 к логиту против шума sigma=1.28.
    pop_z_clip: float = 2.0
    #: Обрезка трендовости (в единицах «раз от равномерной доли»). Член
    #: трендовости сам себя разгоняет: больше трафика -> выше доля -> ещё
    #: больше трафика. Обрезка обрывает петлю.
    pd_rel_clip: float = 8.0
    activity_sigma: float = 0.9   # лог-нормальный разброс активности

    #: калибровка воронки: базовые вероятности эскалации при нулевом сигнале
    p_cart_base: float = 0.11
    p_purchase_given_cart: float = 0.36
    escalate_scale: float = 0.35
    seed: int = 0

    #: ЦЕНОВОЙ СЕГМЕНТ ОСТАЁТСЯ ЧИСТЫМ ШУМОМ, намеренно.
    #: Это отрицательный контроль полигона: атрибут есть в схеме, компилятор
    #: обязан перебрать мета-пути через него — и обязан НЕ найти там сигнала.
    #: Полигон, где все атрибуты полезны, не отличил бы находку от угадывания.

    def ground_truth(self) -> dict:
        return {
            "tau_cat_days": self.tau_cat_days,
            "tau_brand_days": self.tau_brand_days,
            "b_brand": self.b_brand,
            "tau_item_days": self.tau_item_days,
            "tau_pop_days": self.tau_pop_days,
            "b_cat": self.b_cat,
            "b_item": self.b_item,
        }


@numba.njit(cache=True, nogil=True)
def _simulate(
    t_ev, u_ev, cum_pop,
    item_cat, n_cats, item_brand, n_brands,
    lam_cat, lam_brand, lam_item, lam_pop, lam_sat,
    b_pop_s, b_pop_d, b_cat, b_brand, b_item, b_sat,
    base_pop_std, n_candidates, n_recent, n_recent_cand,
    rep_cap, pd_rel_clip, p_cart_base, p_pur_given_cart, escalate_scale, seed,
):
    """Порождение (item, etype) для заранее сгенерированных (t, user).

    Состояние:
      aff_cat[u, c]    интерес пользователя к категории, затухание lam_cat
      rec_item[u, r]   кольцевой буфер недавних айтемов пользователя
      rec_w[u, r]      вес повтора,    затухание lam_item
      rec_s[u, r]      вес пресыщения, затухание lam_sat (быстрое)
      pop_d[i]         трендовость айтема, затухание lam_pop

    Затухание ленивое: t_touch по пользователю и по айтему.
    """
    np.random.seed(seed)
    n = t_ev.shape[0]
    n_users = int(u_ev.max()) + 1
    n_items = int(cum_pop.shape[0])

    aff_cat = np.zeros((n_users, n_cats), dtype=np.float32)
    aff_brand = np.zeros((n_users, n_brands), dtype=np.float32)
    t_user = np.zeros(n_users, dtype=np.float64)
    rec_item = np.full((n_users, n_recent), -1, dtype=np.int32)
    rec_w = np.zeros((n_users, n_recent), dtype=np.float32)
    rec_s = np.zeros((n_users, n_recent), dtype=np.float32)
    rec_ptr = np.zeros(n_users, dtype=np.int32)
    pop_d = np.zeros(n_items, dtype=np.float32)
    t_item = np.zeros(n_items, dtype=np.float64)
    # глобальный затухающий трафик: нужен, чтобы трендовость была ДОЛЕЙ,
    # а не абсолютным счётчиком. Абсолютный счётчик растёт вместе с общим
    # трафиком и даёт разгон «популярное -> ещё популярнее» без насыщения.
    tot_pop = 0.0
    t_tot = 0.0
    # бегущее среднее силы сигнала: калибрует воронку независимо от
    # абсолютного масштаба логитов
    ema_det = 0.0
    ema_init = False
    logit_cart_base = np.log(p_cart_base / (1.0 - p_cart_base))

    out_item = np.empty(n, dtype=np.int32)
    out_etype = np.empty(n, dtype=np.int8)

    n_slot = n_candidates + n_recent_cand
    logits = np.empty(n_slot, dtype=np.float64)
    cands = np.empty(n_slot, dtype=np.int32)

    for e in range(n):
        t = t_ev[e]
        u = u_ev[e]

        # ---- ленивое затухание состояния пользователя ----
        dtu = t - t_user[u]
        if dtu > 0.0:
            dc = np.exp(-lam_cat * dtu)
            for c in range(n_cats):
                aff_cat[u, c] *= dc
            db = np.exp(-lam_brand * dtu)
            for bb in range(n_brands):
                aff_brand[u, bb] *= db
            di = np.exp(-lam_item * dtu)
            ds = np.exp(-lam_sat * dtu)
            for r in range(n_recent):
                rec_w[u, r] *= di
                rec_s[u, r] *= ds
            t_user[u] = t

        # ---- кандидаты: популярные + ПОДМНОЖЕСТВО недавних, без дублей ----
        m = 0
        for _ in range(n_candidates):
            x = np.random.random()
            lo = 0
            hi = n_items - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cum_pop[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            cands[m] = lo
            m += 1

        added = 0
        for r in range(n_recent):
            if added >= n_recent_cand:
                break
            it = rec_item[u, r]
            if it < 0:
                continue
            dup = False
            for j in range(m):          # дедупликация: иначе один айтем
                if cands[j] == it:      # попадает в кандидаты по разу на слот
                    dup = True
                    break
            if not dup:
                cands[m] = it
                m += 1
                added += 1

        # ---- скоринг ----
        dtt = t - t_tot
        if dtt > 0.0:
            tot_pop *= np.exp(-lam_pop * dtt)
            t_tot = t
        pop_scale = tot_pop if tot_pop > 1.0 else 1.0

        best = -1.0e30
        best_det = 0.0
        pick = 0
        for j in range(m):
            i = cands[j]
            dti = t - t_item[i]
            pd = pop_d[i] * np.exp(-lam_pop * dti) if dti > 0.0 else pop_d[i]
            # доля айтема в текущем трафике, в единицах «раз от равномерной»
            pd_rel = pd * n_items / pop_scale
            if pd_rel > pd_rel_clip:
                pd_rel = pd_rel_clip

            w_rep = 0.0
            w_sat = 0.0
            for r in range(n_recent):
                if rec_item[u, r] == i:
                    w_rep += rec_w[u, r]
                    w_sat += rec_s[u, r]
            if w_rep > rep_cap:        # насыщение: обрывает разгон обратной связи
                w_rep = rep_cap

            det = (b_pop_s * base_pop_std[i]
                   + b_pop_d * np.log1p(pd_rel)
                   + b_cat * np.log1p(aff_cat[u, item_cat[i]])
                   + b_brand * np.log1p(aff_brand[u, item_brand[i]])
                   + b_item * np.log1p(w_rep)
                   - b_sat * np.log1p(w_sat))
            g = -np.log(-np.log(np.random.random() + 1e-12) + 1e-12)
            lg = det + g
            logits[j] = lg
            if lg > best:
                best = lg
                best_det = det
                pick = j
        item = cands[pick]

        # ---- тип события: калиброванная воронка ----
        # Модуляция идёт от ДЕТЕРМИНИРОВАННОЙ части сигнала, центрированной
        # бегущим средним. Поэтому средняя P(cart) держится у p_cart_base
        # независимо от того, какой абсолютный масштаб у логитов.
        if not ema_init:
            ema_det = best_det
            ema_init = True
        else:
            ema_det += 0.001 * (best_det - ema_det)
        z = logit_cart_base + escalate_scale * (best_det - ema_det)
        p_cart = 1.0 / (1.0 + np.exp(-z))
        r1 = np.random.random()
        if r1 < p_cart:
            et = 2 if np.random.random() < p_pur_given_cart else 1
        else:
            et = 0

        out_item[e] = item
        out_etype[e] = et

        # ---- обновление состояния ----
        # вес по типу события: покупка — более сильный сигнал интереса
        w_ev = 1.0 if et == 0 else (1.6 if et == 1 else 2.2)
        aff_cat[u, item_cat[item]] += w_ev
        aff_brand[u, item_brand[item]] += w_ev
        p = rec_ptr[u]
        rec_item[u, p] = item
        rec_w[u, p] = 1.0            # вес повтора не зависит от типа: иначе
        rec_s[u, p] = 1.0            # покупки сами себя разгоняют
        rec_ptr[u] = (p + 1) % n_recent

        dti = t - t_item[item]
        if dti > 0.0:
            pop_d[item] *= np.exp(-lam_pop * dti)
        pop_d[item] += 1.0
        t_item[item] = t
        tot_pop += 1.0

    return out_item, out_etype


def generate(cfg: SyntheticConfig | None = None) -> tuple[EventLog, dict]:
    """Сгенерировать лог. Возвращает (лог, истинные параметры)."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    # атрибуты айтемов
    item_cat = rng.integers(0, cfg.n_cats, cfg.n_items).astype(np.int32)
    item_brand = rng.integers(0, cfg.n_brands, cfg.n_items).astype(np.int32)
    item_price = rng.integers(0, cfg.n_price_buckets, cfg.n_items).astype(np.int32)

    # популярность по Ципфу
    ranks = np.arange(1, cfg.n_items + 1)
    base_pop = 1.0 / ranks ** cfg.zipf_a
    rng.shuffle(base_pop)
    # пул кандидатов: популярность в степени alpha < 1
    p_cand = base_pop ** cfg.cand_pop_alpha
    p_cand /= p_cand.sum()
    cum_pop = np.cumsum(p_cand)
    cum_pop[-1] = 1.0

    # активность пользователей: лог-нормальная
    rate = rng.lognormal(0.0, cfg.activity_sigma, cfg.n_users)
    rate /= rate.sum()

    # суперпозиция пуассоновских процессов: времена + кто именно действует
    t_days = np.sort(rng.uniform(0.0, cfg.span_days, cfg.n_events))
    u_ev = rng.choice(cfg.n_users, size=cfg.n_events, p=rate).astype(np.int32)

    # стандартизованная лог-популярность: слагаемые logit должны быть
    # сопоставимы друг с другом и с gumbel-шумом
    bp = np.log1p(base_pop * cfg.n_items)
    base_pop_std = np.clip((bp - bp.mean()) / (bp.std() + 1e-9),
                           -cfg.pop_z_clip, cfg.pop_z_clip).astype(np.float64)

    ln2 = float(np.log(2.0))
    items, etypes = _simulate(
        t_days, u_ev, cum_pop,
        item_cat, cfg.n_cats, item_brand, cfg.n_brands,
        ln2 / cfg.tau_cat_days, ln2 / cfg.tau_brand_days,
        ln2 / cfg.tau_item_days, ln2 / cfg.tau_pop_days,
        ln2 / cfg.satiation_halflife_days,
        cfg.b_pop_static, cfg.b_pop_decay, cfg.b_cat, cfg.b_brand,
        cfg.b_item, cfg.b_satiation,
        base_pop_std,
        cfg.n_candidates, cfg.n_recent, cfg.n_recent_candidates,
        cfg.rep_cap, cfg.pd_rel_clip, cfg.p_cart_base, cfg.p_purchase_given_cart,
        cfg.escalate_scale, cfg.seed,
    )

    t_sec = (t_days * SEC_PER_DAY).astype(np.int64)
    log = EventLog(
        t=t_sec,
        user=u_ev,
        item=items,
        etype=etypes,
        attrs={
            "cat": item_cat[items],
            "brand": item_brand[items],
            "price_bucket": item_price[items],
        },
        etype_names=["view", "cart", "purchase"],
        n_users=cfg.n_users,
        n_items=cfg.n_items,
        attr_cardinality={
            "cat": cfg.n_cats,
            "brand": cfg.n_brands,
            "price_bucket": cfg.n_price_buckets,
        },
    )
    truth = cfg.ground_truth()
    truth["item_cat"] = item_cat
    return log, truth


# --------------------------------------------------------------------------
# Хирургический полигон: единственный ИЗВЕСТНЫЙ оптимальный полупериод
# --------------------------------------------------------------------------
#
# Зачем нужен отдельный генератор. На богатом полигоне выше замерено, что
# параметр порождения tau_item = 4 дня даёт оптимальный ПРЕДСКАЗАТЕЛЬНЫЙ
# полупериод 12 дней при негативах из популярности и 0.66 дня при негативах
# из истории пользователя. Это не ошибка метода: член склонности в модели
# порождения и цель предсказания — разные функционалы одного процесса,
# поэтому «истинный полупериод» не является свойством данных.
#
# Из-за этого на богатом полигоне нельзя проверить саму МАШИНЕРИЮ: если
# метод нашёл 12 дней вместо 4, непонятно, метод ошибся или постановка.
#
# Здесь метка строится так, что оптимальный полупериод существует и равен
# в точности lambda_star по построению:
#
#     P(y = 1 | u, i, t) = sigmoid(a + b * z(S_{lambda_star}(key, t)))
#
# где S — затухающий счётчик по ключу, z — стандартизация. Никаких других
# зависимостей от времени в метке нет. Если пайплайн не находит здесь
# lambda_star, он сломан; если находит — машинерия верна, и расхождения на
# богатом полигоне объясняются постановкой, а не реализацией.

def generate_known_optimum(
    true_halflife_days: float = 12.0,
    key: tuple[str, ...] = ("user", "cat"),
    n_users: int = 8_000,
    n_items: int = 2_000,
    n_cats: int = 40,
    n_events: int = 400_000,
    n_queries: int = 60_000,
    span_days: float = 240.0,
    beta: float = 1.5,
    intercept: float = -1.0,
    seed: int = 0,
) -> tuple[EventLog, QuerySet, dict]:
    """Лог и запросы, в которых оптимальный полупериод равен true_halflife_days.

    История событий порождается без временной структуры (однородный
    пуассоновский поток со случайными парами) — вся временная зависимость
    метки идёт ровно через один затухающий счётчик с известным ядром.
    """
    from ..dsl import ANY_ETYPE, FeatureSet, Primitive
    from ..engine import FeatureCompiler
    from ..log import QuerySet, build_stream

    rng = np.random.default_rng(seed)
    item_cat = rng.integers(0, n_cats, n_items).astype(np.int32)

    # --- история: без временной структуры, но с неоднородной популярностью,
    #     иначе счётчики по ключу почти не различаются между строками
    ranks = np.arange(1, n_items + 1)
    p_item = 1.0 / ranks ** 0.7
    rng.shuffle(p_item)
    p_item /= p_item.sum()
    p_user = rng.lognormal(0.0, 0.8, n_users)
    p_user /= p_user.sum()

    t_days = np.sort(rng.uniform(0.0, span_days, n_events))
    log = EventLog(
        t=(t_days * SEC_PER_DAY).astype(np.int64),
        user=rng.choice(n_users, n_events, p=p_user).astype(np.int32),
        item=(items := rng.choice(n_items, n_events, p=p_item).astype(np.int32)),
        etype=np.zeros(n_events, dtype=np.int8),
        attrs={"cat": item_cat[items]},
        etype_names=["view"],
        n_users=n_users, n_items=n_items, attr_cardinality={"cat": n_cats},
    )

    # --- запросы на второй половине периода
    tq = np.sort(rng.uniform(span_days * 0.5, span_days, n_queries))
    q_user = rng.choice(n_users, n_queries, p=p_user).astype(np.int32)
    q_item = rng.choice(n_items, n_queries, p=p_item).astype(np.int32)
    queries = QuerySet(
        t=(tq * SEC_PER_DAY).astype(np.int64),
        user=q_user, item=q_item,
        y=np.zeros(n_queries, dtype=np.int8),
        group=np.arange(n_queries, dtype=np.int64),
        attrs={"cat": item_cat[q_item]},
    )

    # --- метка строится из счётчика с ИЗВЕСТНЫМ ядром
    prim = Primitive(key=key, etype=ANY_ETYPE, stat="dcount",
                     halflife_s=true_halflife_days * SEC_PER_DAY)
    S = FeatureCompiler(FeatureSet([prim], []), n_etypes=1, dtype=np.float64,
                        state_dtype=np.float64).transform(
        build_stream(log, queries), verbose=False)[:, 0]
    S = np.nan_to_num(S, nan=0.0)
    z = (S - S.mean()) / (S.std() + 1e-9)
    p = 1.0 / (1.0 + np.exp(-(intercept + beta * z)))
    queries.y = (rng.random(n_queries) < p).astype(np.int8)

    truth = {
        "true_halflife_days": true_halflife_days,
        "key": key,
        "beta": beta,
        "positive_rate": float(queries.y.mean()),
    }
    return log, queries, truth
