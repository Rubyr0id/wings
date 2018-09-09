# -*- coding: utf-8 -*-
# Тут основной код проекта.

from __future__ import division
import numpy as np
import pandas as pd
from datetime import datetime as dt
#import pandas.core.algorithms as algos
from sklearn.base import BaseEstimator, TransformerMixin
#import itertools as it
#import logging
#import sys
from .functions import calc_descriptive_from_vector,split_by_edges,gini_index,calculate_loc_woe,is_integer_func_inner,check_nulls_in_freq
from .optimizer import WingOptimizer
from sklearn.model_selection import check_cv


class WingOfEvidence(BaseEstimator, TransformerMixin):
    """
    Этот класс реализует WoE-расчет для одной переменной
    """
    def __init__(self,variable_name,
                 n_initial=10, n_target=5,
                 spec_values={},
                 bin_size_min=None,
                 bin_size_max=None,
                 print_=True,
                 keep_n_best_variants=0,
                 positive_dependance=None,
                 min_diff=0,
                 max_extrema=None
                 ):
        """
        Инициация класса.
        Пропуски будут автоматически отнесены в отдельную группу "AUTO_MISS"
        Args:
            vector_type (str, "c" для непрерывной переменной и "d" для объектной):
                тип вектора:
                    "c" для непрерывной переменной
                    "d" для дискретной
            n_initial (int):
                Количество стартовых групп для поиска
            n_target (int):
                количество макс. групп для поиска
            spec_values (:obj:`dict`):
                Словарь для отбора специальных значений в переменной.
                Пример: {1:"ONE_GROUP"}
        """
        self.variable_name = variable_name
        self.n_initial = n_initial
        self.n_target = n_target
        self.spec_values = spec_values
        self.bin_size_min = bin_size_min
        self.bin_size_max = bin_size_max 
        self.print_ = print_
        self.keep_n_best_variants = keep_n_best_variants
        self.min_diff = min_diff
        self.positive_dependance = positive_dependance
        self._check_params()
        self.d_cnt = -1
        # словари для категориальных признаков        
        self.int_to_cat = None
        self.cat_to_int = None
        self.max_extrema = max_extrema
        self.optimizer = None
        self.vector_type = 'c'

    def __missing_handler(self, df):
        """
        Функция обрабатывает пропущенные значения.
        Args:
            df (pd.DataFrame):
                DF с колонками: ["X","y"]
        Returns:
            df (pd.DataFrame);
                DF с колонками: ["X","y"] без пропущенных значений.
            miss_woe (dict):
                Возвращает словарь с рассчитаным miss_woe и характеристиками (или None, если миссингов нет).
        """
        loc_df = df.copy()
        missing_mask = loc_df["X"].isnull()
        miss_df = loc_df[missing_mask]
        non_miss_df = loc_df[~missing_mask]
        if len(miss_df) == 0:
            dict_skeleton = {
                "events": None,
                "non_events": None,
                "total": None ,
                "woe": None,
                "local_event_rate": None}
            return (non_miss_df, dict_skeleton)
        else:
            vect = pd.Series([
                miss_df["y"].sum(),
                len(miss_df) - miss_df["y"].sum()],
                index=["events", "non_events"])
            miss_woe = calculate_loc_woe(vect, self.__TOTAL_EVENTS, self.__TOTAL_NON_EVENTS)
            miss_woe_d = {
                "events": vect["events"],
                "non_events": vect["non_events"],
                "total": (vect["events"] + vect["non_events"]),
                "woe": miss_woe,
                "local_event_rate": vect["events"] / (vect["events"] + vect["non_events"])}
            return (non_miss_df, miss_woe_d)

    def __special_handler(self, df):
        """
        Функция обрабатывает специальные значения.
        Args:
            df (pd.DataFrame):
                DF с колонками: ["X","y"]
        Returns:
            df (pd.DataFrame);
                DF с колонками: ["X","y"] без спецзначений
            spec_values_woe (float):
                Словарь со значениями spec_values.
                Если нет spec_values в обучающем Df, вернет None для данного ключа.
        """
        loc_df = df.copy()
        special_mask = loc_df["X"].isin(list(self.spec_values.keys()))
        special_df = loc_df[special_mask]
        non_special_df = loc_df[~special_mask]
        spec_values_woe = {}
        for key in self.spec_values:
            key_df = special_df[special_df["X"] == key]
            if len(key_df) == 0:
                dict_skeleton = {
                    "events": None,
                    "non_events": None,
                    "total": None,
                    "woe": None,
                    "local_event_rate": None}
                spec_values_woe[key] = dict_skeleton
            else:
                vect = pd.Series([
                    key_df["y"].sum(),
                    len(key_df) - key_df["y"].sum()],
                    index=["events", "non_events"])
                key_woe = calculate_loc_woe(vect, self.__TOTAL_EVENTS, self.__TOTAL_NON_EVENTS)
                spec_values_woe[key] = {
                    "events": vect["events"],
                    "non_events": vect["non_events"],
                    "total": (vect["events"] + vect["non_events"]),
                    "woe": key_woe,
                    "local_event_rate": vect["events"] / (vect["events"] + vect["non_events"])}
        return (non_special_df, spec_values_woe)

    def fit(self, X, y, X_test, y_test, nan_vector=None):
        """
        Обучает модель.
        Args:
            X (np.ndarray):
                Одномерный массив с X
            y (np.ndarray):
                Одномерный массив с y
        Логика данной функции:
            1. Проверить входные данные
            1. Обработать пропуски
            2. Обработать спец-значения
            3. Обработать основной массив
        """
        # Проверяем входной вектор на содержимое
        df = self._check_data(X.loc[:,X.columns != 'freq_var'], y)
#        if self.print_ is True:
#            print("Data check success,df size: %s"%df.shape[0])
        if nan_vector is None:
            self.__TOTAL_EVENTS = df["y"].sum()
            self.__TOTAL_NON_EVENTS = X['freq_var'].sum() - self.__TOTAL_EVENTS
        else:
            self.__TOTAL_EVENTS = df["y"].sum()+nan_vector['events']
            self.__TOTAL_NON_EVENTS = X['freq_var'].sum() - df["y"].sum() + nan_vector['non_events']
        if nan_vector is None:
            df, miss_woe = self.__missing_handler(df)
        else:
            miss_woe = {
                    "events": nan_vector["events"],
                    "non_events": nan_vector["non_events"],
                    "total": (nan_vector["events"] + nan_vector["non_events"]),
                    "woe": calculate_loc_woe(nan_vector, self.__TOTAL_EVENTS, self.__TOTAL_NON_EVENTS),
                    "local_event_rate": nan_vector["events"] / (nan_vector["events"] + nan_vector["non_events"])
                    }
        self.miss_woe = miss_woe
        if self.print_ is True:
            print("miss woe: total %s rate %s"% (self.miss_woe['total'],self.miss_woe['local_event_rate']))
        if len(df) == 0:
            # Весь оставшийся набор данных пустой
            # Значит в датасете были только миссинги, fit закончен
            return self
        # Теперь проверяем спецзначения.
        df, spec_values_woe = self.__special_handler(df)
        self.spec_values_woe = spec_values_woe
        if self.print_ is True and len(self.spec_values_woe)>0:
            print("spec woe %s" % self.spec_values_woe)
        if len(df) == 0:
            # после отбора миссингов и спецзначений датасет пуст
            return self
        # подсчёт уникальных значений признака
        self.d_cnt = len(X)
        freq_np = X['freq_var'].values
        if self.print_ is True:
            print("D-values in  clear X: %i" % self.d_cnt)
        # Преобразование категориального признака (замена на числа) для поиска оптимальных сочетаний категорий
        if df["X"].dtype==np.dtype("O"):
            # вывоз фукнции calc_descriptive_from_vector исключительно для расчёта local_event_rate
            categ_df_woe = calc_descriptive_from_vector(df["X"].values,
                                                        freq_np,
                                                        df["y"].values,
                                                        self.__TOTAL_EVENTS,
                                                        self.__TOTAL_NON_EVENTS,
                                                        bin_size_min=None,
                                                        bin_size_max=None,
                                                        )
            # при замене категории на число у вектора значений появляется "направление"
            # числовые значения присваиваются категориям в порядке убывания local_event_rate, чтобы процедуре было легче найти "похожие" по этому признаку категории и склеить их в один WOE-бакет
            self.int_to_cat = categ_df_woe.sort_values(['local_event_rate','total'],ascending=False).reset_index()['grp'].to_dict()
            self.cat_to_int = dict([(v,k) for k,v in self.int_to_cat.items()])
            df["X"] = df["X"].map(self.cat_to_int).astype(int)

        
        if df["X"].dtype == 'float':
            # Если все значения int, то преобразуем в int,
            # чтобы получились "целые границы", если бининг будет по уникальным значениям, а не квантилям.
            if df["X"].apply(is_integer_func_inner).all():
                df["X"] = df["X"].astype(int)
        # если уникальных значений слишком мало, то нет смысла делать лишние проверки, поэтому работаем с признаком как с дискретным
        if self.d_cnt < 3:
            self.vector_type = 'd'
        # если уникальных значений меньше чем целевое кол-во WOE-бакетов
        elif (self.d_cnt < self.n_target):
            self.n_initial = self.d_cnt                
            self.n_target = self.d_cnt
            if self.print_ is True:
                print("Подменяем начальное и целевое разбиение")
        # если уникальных значений меньше чем начальное кол-во WOE-бакетов
        elif (self.d_cnt < self.n_initial):
            self.n_initial = self.d_cnt
        
        if self.vector_type == "c":
            #######################################################
            #  тут рассчитываем для непрерывной переменной
            #######################################################
            X,y = df["X"].values, df["y"].values
            self.optimizer = WingOptimizer(X,freq_np,y,
                                           total_events=self.__TOTAL_EVENTS,
                                           total_non_events=self.__TOTAL_NON_EVENTS,
                                           n_initial=self.n_initial,
                                           n_target=self.n_target,
                                           bin_size_min=self.bin_size_min,
                                           bin_size_max=self.bin_size_max,
                                           print_=self.print_,
                                           keep_n_best_variants=self.keep_n_best_variants,
                                           dtype = str(X.dtype),
                                           min_diff=self.min_diff,
                                           positive_dependance=self.positive_dependance,
                                           max_extrema=self.max_extrema)
            self.optimal_edges,best_gini = self.optimizer.optimize()
            if best_gini is None:
                if self.print_ is True:
                    print("Binning process failed")
                self.cont_df_woe = None
                
            else:
                if self.print_ is True:
                    print("Optimal edges found: %s"%self.optimal_edges.tolist())
#                    print("With gini: %0.4f"%best_gini)
                bins = split_by_edges(X,self.optimal_edges)
                self.cont_df_woe = calc_descriptive_from_vector(bins,freq_np,y,
                                                                self.__TOTAL_EVENTS,self.__TOTAL_NON_EVENTS,
                                                                self.bin_size_min,
                                                                self.bin_size_max,
                                                                min_diff=self.min_diff)
                self.optimal_edges_dict = self._generate_edge_dict(self.optimal_edges)
                self.wing_id_dict = self.cont_df_woe["woe"].to_dict()
        else:
            #######################################################
            #  тут рассчитываем для дискретной переменной
            #######################################################
            discrete_df = df
            discrete_df["woe_group"] = discrete_df["X"]
            self.discrete_df_woe = calc_descriptive_from_vector(discrete_df["woe_group"].values,
                                                                freq_np,
                                                                discrete_df["y"].values,
                                                                self.__TOTAL_EVENTS,self.__TOTAL_NON_EVENTS,
                                                                self.bin_size_min,
                                                                self.bin_size_max)
            if self.discrete_df_woe is None:
                if self.print_ is True:
                    print("Binning process failed")

        return self

    def transform(self, X, y=None):
        if y is None:
            # bugfix for compatability
            y = pd.Series([1 for i in range(len(X))])
        df = self._check_data(X, y)
        # fill miss
        miss_df = df[pd.isnull(df["X"])].copy()
        miss_df["woe_group"] = "AUTO_MISS"
        miss_df["woe"] = self.miss_woe["woe"]
#        miss_df["woe"] = None
        #######################################################
        # TODO: Расписать что тут происходит
        #######################################################
        spec_df = df[df["X"].isin(self.spec_values)].copy()
        spec_df["woe_group"] = spec_df["X"].apply(lambda x: self.spec_values.get(x))
        spec_df["woe"] = spec_df["X"].apply(lambda x: self.spec_values_woe.get(x).get("woe"))
        # fill dat
        flt_conc = (~pd.isnull(df["X"]) & (~df["X"].isin(self.spec_values)))
        clear_df = df[flt_conc].copy()
        if self.vector_type == "c":
            #######################################################
            # быстрый фикс ошибки в том случае, когда opt
            # не рассчитан
            #######################################################
            if hasattr(self,"optimal_edges"):
                if self.int_to_cat is not None:
                    clear_df["X"] = clear_df["X"].map(self.cat_to_int)
                clear_df["woe_group"] = split_by_edges(clear_df["X"], self.optimal_edges)
                clear_df["woe"] = clear_df["woe_group"].apply(lambda x: self.wing_id_dict[x])
            else:
                clear_df["woe_group"] = "NO_GROUP"
                clear_df["woe"] = None
        else:
            if hasattr(self, "discrete_df_woe"):
                clear_df["woe_group"] = clear_df["X"]
                clear_df["woe"] = pd.merge(clear_df, self.discrete_df_woe, left_on="woe_group", right_index=True, how="inner")["woe"]
            else:
                clear_df["woe_group"] = "NO_GROUP"
                clear_df["woe"] = None
        miss_df["woe_group"] = miss_df["woe_group"].astype(str)
        spec_df["woe_group"] = spec_df["woe_group"].astype(str)
        clear_df["woe_group"] = clear_df["woe_group"].astype(str)
        full_transform = pd.concat([miss_df, spec_df, clear_df], axis=0)  # ["woe"]
        #######################################################
        # TODO: Расписать что тут происходит + алго выбора
        #######################################################
        miss_wing_selector = [self.miss_woe["woe"]]
        spec_wing_selector = [sub_d.get("woe") for sub_d in self.spec_values_woe.values()]
        if self.vector_type == "c":
            if hasattr(self,"wing_id_dict"):
                grpd_wing_selector = list(self.wing_id_dict.values())
            else:
                grpd_wing_selector = [None]
        else:
            grpd_wing_selector = list(self.discrete_df_woe["woe"].values)
        allv_wing_selector = miss_wing_selector+spec_wing_selector+grpd_wing_selector
        allv_wing_selector_flt = [v for v in allv_wing_selector if v is not None]
        max_woe_replacer = np.min(allv_wing_selector_flt)
        full_transform["woe"] = full_transform["woe"].fillna(max_woe_replacer)
        # full_transform = full_transform.sort_index()
        return full_transform

    def get_wing_agg(self, only_clear=False):
        """
        Shows result of WoE fitting as table bins,woe,iv
        Returns:
            woe_df (pd.DataFrame): data frame with WoE fitter parameters
        """
        if only_clear:
            if self.vector_type == "c":
                if self.cont_df_woe is None:
                    return None
                cont_df_woe_loc = self.cont_df_woe.copy()
                if self.cat_to_int is None:
                    cont_df_woe_loc.index = [self.optimal_edges_dict[v] for v in cont_df_woe_loc.index]
                else:
                    cont_df_woe_loc.index = [','.join([v for k,v in self.int_to_cat.items()
                                             if k>=self.optimal_edges_dict[vv][0] 
                                             and k<self.optimal_edges_dict[vv][1]]) for vv in cont_df_woe_loc.index]
                return cont_df_woe_loc
            else:
                return self.discrete_df_woe
        if any(self.miss_woe.values()): #and self.miss_woe
            miss_wect = pd.DataFrame.from_dict(self.miss_woe,orient='index').T
        else:
            miss_wect = pd.DataFrame(columns=["events", "non_events", "woe", "total", "local_event_rate"])
#        if self.spec_values_woe:
#            spec_v_df = pd.DataFrame.from_dict(self.spec_values_woe,orient='index').T
#        else:
#            spec_v_df = pd.DataFrame(columns=["events", "non_events", "woe", "total", "local_event_rate"])
        if self.vector_type == "c":
            miss_wect = miss_wect[['events', 'non_events', 'total', 'woe', 'local_event_rate']]
#            spec_v_df = spec_v_df[['events', 'non_events', 'total', 'woe', 'local_event_rate']]
            if self.cont_df_woe is None:
                return None
            cont_df_woe_loc = self.cont_df_woe.copy()
            # Если словарь, то в качестве индекса будет список категорий
            if self.cat_to_int is None:
                cont_df_woe_loc.index = [self.optimal_edges_dict[v] for v in cont_df_woe_loc.index]
            else:
                cont_df_woe_loc.index = [','.join([v for k,v in self.int_to_cat.items() 
                                          if k>=self.optimal_edges_dict[vv][0] 
                                          and k<self.optimal_edges_dict[vv][1]]) for vv in cont_df_woe_loc.index]
            full_agg = pd.concat([miss_wect, 
#                                  spec_v_df, 
                                  cont_df_woe_loc], axis=0)
        else:
            miss_wect = miss_wect[['events', 'non_events', 'total', 'woe', 'local_event_rate']]
#            spec_v_df = spec_v_df[['events', 'non_events', 'total', 'woe', 'local_event_rate']]
            full_agg = pd.concat([miss_wect, 
#                                  spec_v_df, 
                                  self.discrete_df_woe], axis=0)
        return full_agg

    def get_global_gini(self,only_clear=False):
        woe_df = self.get_wing_agg(only_clear)
        if woe_df is None:
            return None
        woe_df = woe_df.sort_values(by="local_event_rate", ascending=False)
        gini_index_value = gini_index(woe_df["events"].values, woe_df["non_events"].values)
        return gini_index_value

    def _check_params(self):
        """
        This method checks parameters in __init__, raises error in case of errors
        Args:
            None
        Returns None
        """
        if self.n_initial < self.n_target:
            raise ValueError("Number of target groups higher than pre-binning groups")
        if self.n_target <= 1:
            raise ValueError("Set more target groups to search optimal parameters")

    def _check_data(self, X, y):
        """
        Should raise some error if any test is not OK, else do nothing
        Args:
            X (numpy.ndarray): numpy array of X
            y (numpy.ndarray): numpy array of y
        Returns:
            None if everything is ok, else raises error
        """
        if (X.size != y.size):
            raise ValueError("y-size ( %i ) doesn't match X size ( %i )" % (y.size, X.size))
        try:
            X = np.array(X).ravel()
            y = np.array(y).ravel()
        except:
            raise ValueError("X or Y vector cannot by transformed to np.array")
        common_df = pd.DataFrame(np.array([X, y]).T, columns=["X", "y"])
        return common_df

    def _generate_edge_dict(self, edges):
        edges_dict = {}
        for idx, (low, high) in enumerate(zip(edges, edges[1:])):
            edges_dict[idx + 1] = (low, high)
        return edges_dict

    def plot_woe(self):
        """
        Creates woe plot for data in woe_df
        """
        import matplotlib.pyplot as plt
        woe_df = self.get_woe()
        f, ax = plt.subplots(1, 1)
        p1 = woe_df.plot(kind="bar", x=woe_df.index, y="events", ax=ax, sharex=True, figsize=(20, 10), edgecolor="k")
        p2 = woe_df.plot(kind="line", x=woe_df.index, y="woe", ax=ax, secondary_y=True, style="o-", lw=3, c="r", ms=10)
        return (p1, p2)


class WingsOfEvidence(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_apply="all",
                 n_initial=10, n_target=5,
                 mass_spec_values={},
                 bin_size_min=None,
                 bin_size_max=None,
                 print_=True,
                 keep_n_best_variants=0,
                 positive_dependance=None,
                 min_diff=0,
                 max_extrema=None,
                 cv=1
                 ):
        """
        Этот класс реализует расчет WoE по многим переменным
        :Args:
            columns_to_apply ("all",list): список колонок, к которым нужно применить преобразование. Если задано "all", рассчитает WoE по всем.
            n_initial (10,int): Число групп для инициирующего разбиения
            n_target (4,int): Число групп для целевого разбиения
            mass_spec_values ({},dict): Словарь спецзначений для колонок, пример: {"col1":{0:"ZERO"}}
            bin_size_min (None,float): Ограничение на минимальный размер WOE-бакета каждого признака, если <1, то обрабатывается как процент, иначе как кол-во наблюдений
            bin_size_max (None,float): Ограничение на максимальный размер WOE-бакета каждого признака
            print_ (True,bool): Печать промежуточных сообщений
            keep_n_best_variants (0,int): сколько суб-оптимальных вариантов WOE-разбиения сохранять для каждого признака
            positive_dependance (None,bool): Ограничение на строго положительную(True) или отрицательную(False) зависимость между признаком и его WOE-трансформацией. В данный момент не работает.
            min_diff (0,int): Ограничение на минимальную разность между соседними WOE-бакетами по target_rate
            max_extrema (None,int): "Степень нелинейности". Максмальное кол-во экстремумов в трансформированном векторе относительно значений признака.
                0 - монотонность признака и WOE, 
                1 - U-образная зависимость разрешена, 
                2 - S-образная зависимость разрешена,
                3 - зависимость между признаком и WOE описывается полиномом 3й степени
                и т.д.
            cv (None,int) - кол-во "фолдов" при кроссвалидации. None - обучение без кроссвалидации.
        """
        self.columns_to_apply = columns_to_apply
        self.mass_spec_values = mass_spec_values
        self.n_initial = n_initial
        self.n_target = n_target
        self.bin_size_min = bin_size_min
        self.bin_size_max = bin_size_max 
        self.print_ = print_
        self.keep_n_best_variants = keep_n_best_variants
        self.min_diff = min_diff
        self.positive_dependance = positive_dependance
        self.max_extrema = max_extrema
        self.cv = cv

    def fit(self, X, y):
        """
        This class fits onefeature woe for each column in columns_to_apply
        Args:
            X (pd.DataFrame): pandas dataframe with X values
            y (pd.Series): pandas series with target value
        Returns:
            self
        """
        if self.columns_to_apply == "all":
            self.columns_to_apply = X.columns
        self.fitted_wing = {}
        self.gini_dict = {}
        self.duration_dict = {}
        self.error_columns = []
        
        cv = check_cv(self.cv, y, classifier=True)
        folds = list(cv.split(X, y))
                
        for learn_idx, test_idx in folds:
            X_learn = X[learn_idx]
            y_learn = y[learn_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            len_df = len(X_learn)
            
            for column in self.columns_to_apply:
                if self.bin_size_min is not None and self.bin_size_min < 1:
                    bin_size_min = self.bin_size_min*len_df
                else:
                    bin_size_min = self.bin_size_min
                if self.bin_size_max is not None and self.bin_size_max < 1:
                    bin_size_max = self.bin_size_max*len_df
                else:
                    bin_size_max = self.bin_size_max
                start_timestamp = dt.now().replace(microsecond=0)
                if self.print_ is True:
                    print("==="*20)
                    print("Working with variable: %s"%column)
                column_dict = self.mass_spec_values.get(column)
                if not column_dict:
                    spec_values = {}
                else:
                    spec_values = column_dict
                    
                wing = WingOfEvidence(variable_name=column,
                                      n_initial=self.n_initial,
                                      n_target=self.n_target,
                                      spec_values=spec_values,
                                      bin_size_min=bin_size_min,
                                      bin_size_max=bin_size_max,
                                      print_=self.print_,
                                      keep_n_best_variants=self.keep_n_best_variants,
                                      min_diff=self.min_diff,
                                      positive_dependance=self.positive_dependance,
                                      max_extrema=self.max_extrema
                                      )
                try:
                    Xf = pd.concat([X_learn[column],y_learn],axis=1).groupby(column)[y_learn.name].agg(['size','sum']).reset_index()
                    Xf.columns = [column,'freq_var',y_learn.name]
                    wing.fit(Xf[[column,'freq_var']], Xf[y_learn.name],X_test,y_test)
                    self.fitted_wing[column] = wing
                    self.gini_dict[column] = wing.get_global_gini()
                    self.duration_dict[column] = (dt.now().replace(microsecond=0)-start_timestamp,wing.d_cnt,X_learn[column].dtype)
                    if self.print_ is True:
                        print("With gini: %0.4f"%self.gini_dict[column],'duration,',self.duration_dict[column][0])
                except Exception as e:
    #                self.gini_dict[column] = None
                    self.error_columns.append(column)
                    self.duration_dict[column] = (dt.now().replace(microsecond=0)-start_timestamp,wing.d_cnt,X_learn[column].dtype)
                    if self.print_ is True:
                        print("Got error: %s" % e,'duration,',self.duration_dict[column][0])
            return self
    
    def transform(self, X, y=None, prefix=True):
        result_frame = pd.DataFrame()
        orig_cols = X.columns
        list_of_valid_cols = [x for x in self.columns_to_apply if x in orig_cols]
        for column in list_of_valid_cols:
            if column not in self.error_columns:
                if self.gini_dict[column] is not None:
                    woe_transformer = self.fitted_wing[column]
                    woe_values = woe_transformer.transform(X[column],y)
                    if prefix:
                        column = "WOE_%s" % column
                    result_frame[column] = woe_values["woe"]
                    if y is not None:
                        result_frame[y.name] = woe_values["y"]
                        
        result_frame = result_frame.sort_index()
        result_frame.index = X.index
        return result_frame

    def get_gini_vector(self):
        gini_series = pd.Series(self.gini_dict)
        return gini_series


if __name__ == "__main__":
    pass