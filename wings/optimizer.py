# -*- coding: utf-8 -*-
#TODO: В файл вынести алгоритмы оптимизации
from .functions import make_edges,generate_combs,check_variant,add_infinity,split_by_edges


class WingOptimizer:
    def __init__(self,X,freq_np,y,
                 total_events,total_non_events,n_initial,n_target,
                 bin_size_min=None,
                 bin_size_max=None,
                 print_=True,
                 keep_n_best_variants=0,
                 min_diff=0,
                 positive_dependance = None,
                 dtype='float',
                 max_extrema = None):
        """
        :param n_initial (int):
            С какого значения инициируем разбиение
        :param n_target (int):
            каков размер макс. групп
        """
        self.X = X
        self.y = y
        self.freq_np = freq_np
        self.n_initial = n_initial
        self.n_target = n_target
        self.total_events = total_events
        self.total_non_events = total_non_events
        self.bin_size_min = bin_size_min
        self.bin_size_max = bin_size_max
        self.print_ = print_
        self.keep_n_best_variants = keep_n_best_variants
        self.next_best = []
        self.dtype = dtype
        self.min_diff = min_diff
        self.positive_dependance = positive_dependance
        self.max_extrema = max_extrema
        
    def optimize(self):
        """
        Класс инициирует основную логику.
        :return:
         opt_edges:
            Оптимально разбитые границы
        """
        self.init_edges = self._initial_split()
        optimization_result = self._search_optimals()
        return optimization_result

    def _initial_split(self):
        """
        Рассчитывает инициирующие границы
        """
        return make_edges(self.X,self.n_initial,self.dtype)

    def _search_optimals(self):
        all_edge_variants = generate_combs(self.init_edges[1:-1],self.n_target)
        mono_variants = []
        for edge_variant in all_edge_variants:
            edge_variant = add_infinity(edge_variant)
            bins = split_by_edges(self.X,edge_variant)
            is_mono,gini = check_variant(bins,self.freq_np,self.y,
                                               total_events=self.total_events,
                                               total_non_events=self.total_non_events,
                                               bin_size_min=self.bin_size_min,
                                               bin_size_max=self.bin_size_max,
                                               min_diff=self.min_diff,
                                               positive_dependance=self.positive_dependance,
                                               max_extrema=self.max_extrema
                                               )
            if is_mono:
                mono_variants.append((edge_variant,gini))
        if not mono_variants:
            if self.print_ is True:
                print("No valid split found")
            optimization_result = None,None
        else:
            if self.print_ is True:
                print("total: %i, mono: %i"%(len(all_edge_variants),len(mono_variants)))
            optimization_result = sorted(mono_variants, key=lambda x: x[1],reverse=True)
            if self.keep_n_best_variants>0:
                for bining_variant in optimization_result[:self.keep_n_best_variants]:
                    self.next_best.append(bining_variant)
            optimization_result = optimization_result[0]
        return optimization_result