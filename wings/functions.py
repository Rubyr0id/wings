# -*- coding: utf-8 -*-
"""
TODO: В файл вынести статические функции, которые будет необходимо оптимизировать на C
Какие функции вообще есть и какие выносим:
make_edges
generate_combs
generate_layer_variant
add_infinity
check_extremum (old name - check_mono)
split_by_edges
gini_index
calc_descriptive_from_df
"""
import numpy as np
import pandas as pd
from itertools import combinations

def is_integer_func_inner(x):
    '''
    True - если объект целое число
    Применяется к "очищенному" от None'ов вектору,
    т.к. наличие None не позволяет привести тип вектора из 'float' к 'int'
    '''
    return (x).is_integer()

def make_edges(X,
               cuts,
               dtype='float',
               unique=True):
    """
    Создаем первичное разбиение на границы
    Args:
        X (numpy.ndarray): массив X
        cuts (int): количество групп
    Returns:
        edges (np.array): массив границ
    """
    needs_unique = False
#    edges_space = np.linspace(0, 1, num=cuts)
    # Делаем проверку потому, что нарезка может быть неудачной из за слишком большого перевеса
    try:
        splits, edges = pd.qcut(X, q=cuts, retbins=True)
    except:
#        print("Too oversampled dataset for qcut, will be used only unique values for splitting")
        needs_unique = True
    if needs_unique:
        splits, edges = pd.qcut(np.unique(X), q=cuts, retbins=True)
#    print(dtype[:3])
    if dtype[:3] == 'int':
        edges = np.ceil(edges)
    edges = np.array([-np.inf] + list(edges[1:-1]) + [np.inf])
    if unique:
        edges = np.unique(edges)
    return edges

def generate_combs(vector, k, k_start=1):
    """
    Генерирует перестановки в виде:
        C(n,1) + C(n,2) + ... + C(n,k)
    Args:
        vector (np.array):
            Вектор для генерации перестановок
        k (int):
            Макс. размер перестановок
        k_start (int):
            С какого k начинать.
    """
    collector = []
    for r in range(k_start, k + 1):
        variants = [el for el in combinations(vector, r)]
        collector.append(variants)
    collector = sum(collector, [])
    return collector

def generate_layer_variant(df, layer_edges, pre_edges):
    new_layer = _combine_vector(layer_edges, pre_edges)
    layer_edges_flt = [variant for variant in new_layer if _calc_bins_and_check_mono(df, variant)]
    return layer_edges_flt

def combine_vector(edge_list, base_edges):
    collec_layer = []
    for vect in edge_list:
        high_vect = np.max(vect)
        vect_typ = list(vect)
        idx_v = base_edges > high_vect
        subselected_max = base_edges[idx_v]
        for v in subselected_max:
            if v != np.inf:
                new_v = vect_typ + [v]
                collec_layer.append(tuple(new_v))
    return collec_layer

def calc_bins_and_check_mono(df,bins):
    layer_bins = _split_by_edges(df["X"], _add_infinity(bins))
    df["bins"] = layer_bins
    variant_woe_vector = _calc_descriptive_from_df(df, "bins")["woe"]
    if _check_mono(variant_woe_vector) and _check_:
        return True
    else:
        return False

def add_infinity(vector):
    """
    Adds inf values to vector
    Args:
        vector (np.array):
            Object to add infs
    Returns:
        vector (np.array):
            object with inf
    """
    inf_vector = np.concatenate([[-np.inf], list(vector), [np.inf]])
    return inf_vector

def check_extremum(vector,vector_rate,min_diff=0,positive_dependance=None,max_extrema=None):
    """
    This function checks:
        if WOE-vector is monotonic,
        if the requirements for a minimim diference between buckets are held
    Woe decrease when target rate increase in current realisation
    So when mono_inc = True, then rate is decreasing, which means that dependance is negative
    
    Args:
        vector (np.array): Vector Array of data
    Returns:
        is_mono (bool)
    """
    diffs = np.diff(vector)
    mono_inc = diffs>0
    if max_extrema is None:
        valid_result = True
    else:
        valid_result = max_extrema >= sum(np.diff(mono_inc))
    if min_diff==0 or not valid_result:
        return valid_result
#    diffsr = np.diff(vector_rate)
#    if np.all(mono_inc):
#        relative_diffs = -diffsr / vector_rate[:1]
#    else:
#        relative_diffs = diffsr / vector_rate[1:]
#    return relative_diffs.min()<min_diff
    # проверка на минимально допустимое отличие между соседними бакетами по target_rate
    diffsr = np.diff(vector_rate)
    if np.all(mono_inc):
        return (-diffsr).min()>min_diff
    else:
        return (diffsr).min()>min_diff


def check_nulls_in_freq(X,y,column):
    '''
    X - pd.DataFrame, array of features
    y - pd.Series, tartet variable
    column - feature name
    ***********************************
    return - number of nulls and amount of "events" in this group
    '''
    do_nulls_exist = X[column].isnull()
    if do_nulls_exist.any():
        Xf_null = pd.concat([do_nulls_exist, y],axis=1).groupby(column).agg(['size','sum']).loc[True]
        Xf_null[0] = Xf_null[0] - Xf_null[1]
        Xf_null.index = ['non_events','events']
    else:
        Xf_null = None
    return Xf_null

def check_rate_diff(vector,min_rate_diff=0):
    diffs = np.diff(vector)
    return diffs

def split_by_edges(vector, edges):
    """
    Splits input vector by edges and returns index of each value
    Args:
        vector (np.array): array to split
        edges (np.array): array of edges
    Returns:
        bins: (np.array): array of len(vector) with index of each element
    """
    # bins = np.digitize(vector,edges,right=True)
    bins = np.digitize(vector, edges)
    return bins

def calculate_loc_woe(vect, events, non_events):
    """
    Calculates woe in bucket
    Args:
        vector (pd.Series): Vector with keys "events" and "non_events"
        event (int): total amount of "event" in frame
        non_events (int): total amount of "non-event" in frame
    """
    t_events = np.float(vect["events"]) / np.float(events)
    t_non_events = np.float(vect["non_events"]) / np.float(non_events)
    if t_non_events == 0:
        t_non_events = 0.0001
    if t_events == 0:
        t_events = 0.0001
    return np.log(t_non_events / t_events)

def gini_index(events, non_events):
    """
    Calculates Gini index in SAS format
    Args:
        events (np.array): Vector of events group sizes
        non_events (np.array): Vector of non-event group sizes
    Returns:
        Gini index (float)
    """
#    p1 = float(2 * sum(events[i] * sum(non_events[:i]) for i in range(1, len(events))))
#    p2 = float(sum(events * non_events))
#    p3 = float(events.sum() * non_events.sum())
    p1 = float(2 * sum(events[i] * non_events[:i].sum() for i in range(1, len(events))))
    p2 = float((events * non_events).sum())
    p3 = float(events.sum() * non_events.sum())
    if p3 == 0.0:
        return 0
    else:
        coefficient = 1 - ((p1 + p2) / p3)
        index = coefficient * 100
        return index


def calc_descriptive_from_vector(bins,
                                 freq_np,
                                 y,
                                 total_events,
                                 total_non_events,
                                 bin_size_min,
                                 bin_size_max,
                                 min_diff=0):
    """
    Calculates IV/WoE + other descriptive data in df by grouper column
    Args:
        df (pd.DataFrame): dataframe with vectors X,y
        grouper (str): grouper of df
    Returns:
        woe_df with information about woe, lre and other.
    """
    df = pd.DataFrame(np.array([bins,y,freq_np]).T,columns=["grp","y","freq"])
    tg_all = df.groupby("grp")["freq"].sum()
    if bin_size_min is not None:
        if (tg_all<bin_size_min).any():
            return None
    if bin_size_max is not None:
        if (tg_all>bin_size_max).any():
            return None
    tg_events = df.groupby("grp")["y"].sum()
    tg_non_events = tg_all - tg_events
    woe_df = pd.concat([tg_events, tg_non_events, tg_all], axis=1)
    woe_df.columns = ["events", "non_events", "total"]
    woe_df["woe"] = woe_df.apply(lambda row: calculate_loc_woe(row, total_events, total_non_events), axis=1)
    woe_df["local_event_rate"] = woe_df["events"] / tg_all
    return woe_df

def check_variant(bins,
                  freq_np,
                  y,
                  total_events,
                  total_non_events,
                  bin_size_min,
                  bin_size_max,
                  min_diff=0,
                  positive_dependance=None,
                  max_extrema=None
                  ):
    """
    Функция разбивает вектор X по edges
    Считает WoE по разбитым группам
    Проверяет размер каждой группы
    Проверяет является ли оно монотонным
    Если нет - gini=None
    Если да - считает gini
    :param bins:
        Вектор примененных границ групп (с бесконечностями по краям)
    :param X:
        Вектор X для разбиения
    :param y:
        Вектор y для расчета gini
    :param bin_size_min:
        Нижний предел для размера бина
    :param bin_size_max:
        Верхний предел для размера бина
    :return:
        edges:
            Исходный набор границ
        is_mono:
            Является ли разбиение монотонным
        gini:
            Если is_mono=False, возращает None
            Если is_mono=True, возвращает значение Gini Index
    """
    # Предварительная проверка на размер бина делается внутри calc_descriptive_from_vector, чтобы не делать лишних вычислений
    wdf = calc_descriptive_from_vector(bins,freq_np,y,total_events,total_non_events,bin_size_min,bin_size_max)
    if wdf is None:
        return False,None
    if check_extremum(wdf["woe"],wdf["local_event_rate"],min_diff,positive_dependance,max_extrema):
        wdf = wdf.sort_values(by="local_event_rate",ascending=False)
        gini_index_value = gini_index(wdf["events"].values, wdf["non_events"].values)
        return True,gini_index_value
    else:
        return False,None

def optimize_edges(clear_df, pre_edges, optimizer):
    """
    Here we optimize edges to find best WoE split
    Args:
        clear_df (pd.DataFrame):
            dataframe of clear values
        pre_bins (np.array):
            array of pre bins
        pre_edges (np.array):
            array of pre edges,
        optimizer (str):
            "full-search" - full search in all combs
            "adaptive" - adaptive search
    Returns:
        optimal_edges (np.array): optimal edges split
    Algo def goes here:
    1. Find all combinations of edges
    2. Generate all edges
    """
    # first - create pre-bins and calculate woe for this
    X_vect = clear_df["X"].values
    pre_bins = self._split_by_edges(X_vect, pre_edges)
    pre_edges_dict = self._generate_edge_dict(pre_edges)
    if self.print_ is True:
        print("Pre edges dict:")
        print(pre_edges_dict)
    pre_bins_dict = pd.Series(pre_bins).apply(lambda x: pre_edges_dict[x])
    if self.print_ is True:
        print("Initial binning:")
        print(pre_bins_dict.value_counts().sort_index())
    self.pre_edges = pre_edges
    clear_df_loc = clear_df.copy()
    clear_df_loc["bins"] = pre_bins
    self.pre_woe_df = self._calc_descriptive_from_df(clear_df_loc, "bins")
    # second - check for monotonical - if pre_edges enougth - return pre_edges, else - search for optimized
    if self._check_mono(self.pre_woe_df["woe"]):
        if self.print_ is True:
            print("Optimal edges found in pre-binning stage")
        optimal_edges = pre_edges
        if lalala:
            if self.print_ is True:
                print("Searching edges via adaptive search algo")
            #######################################################
            # Алгоритм делает следующее:
            # 1. Генерирует перестановки первого уровня
            # 2. Генерирует перестановки второго уровня
            # Для перестановок второго уровня делаем отбор моно
            # Для каждой моно делаем проверку на размер бина
            # Для каждой из выбранных перестановок добавляем новые
            # Повторяем из раза в раз, отбирая моно
            #######################################################
            f1_layer = [el for el in it.combinations(pre_edges[1:-1], 1)]
            f1_layer_flt = [variant for variant in f1_layer if self._calc_bins_and_check_mono(clear_df_loc, variant)]
            f2_layer = [el for el in it.combinations(pre_edges[1:-1], 2)]
            f2_layer_flt = [variant for variant in f2_layer if self._calc_bins_and_check_mono(clear_df_loc, variant)]
            layers_collector = [f1_layer_flt, f2_layer_flt]
            iterator_layer = f2_layer_flt
            for i in range(self.n_target):
                lv = self._generate_layer_variant(clear_df_loc, iterator_layer, pre_edges)
                iterator_layer = lv
                layers_collector.append(lv)
                if self.print_ is True:
                    print("Total variants at level: %i -local: %i" % (i, len(iterator_layer)))
            layers_collector = sum(layers_collector, [])
            final_keeper = []
            for mono_variant in layers_collector:
                if self._calc_bins_and_check_mono(clear_df_loc, mono_variant):
                    mono_variant = self._add_infinity(mono_variant)
                    mono_bins = self._split_by_edges(clear_df_loc["X"], mono_variant)
                    clear_df_loc["bins"] = mono_bins
                    desc_df = self._calc_descriptive_from_df(clear_df_loc, "bins")
                    desc_df = desc_df.sort_values(by="local_event_rate", ascending=False)
                    gini_index = self._gini_index(desc_df["events"].values, desc_df["non_events"].values)
                    final_keeper.append((mono_variant, gini_index))
            # for el in sorted(final_keeper, key=lambda x: x[1]): print(el)
            best_variant = sorted(final_keeper, key=lambda x: x[1])[-1]
            optimal_edges, gini_index_best = best_variant
            if self.print_ is True:
                print("Got best Gini: %0.3f at variant %s" % (gini_index_best, optimal_edges))
    return optimal_edges


if __name__ == "__main__":
    print("Non executable module")