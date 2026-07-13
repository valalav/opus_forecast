"""Dashboard constants module."""

ALL_MODELS = [
    "Ridge",
    "Ridge_Ext",
    "Bayes_Ridge",
    "ElasticNet",
    "Huber",
    "Ridge_Shock",
    "Ridge_Shock_Roll24",
    "Ridge_ProdProxy",
    "Ridge_AsymERPT",
    "Ridge_Macro",
    "Rolling_Ridge",
    "NGBoost",
    "NGBoost_Shock",
    "BVAR",
    "SARIMA",
    "LightGBM",
    "Prophet",
    "ETS",
    "EBM",
    "CatBoost",
    "Subcomp",
    "Subcomp_Multi",
    "Micro",
    "Micro_SM",
    "Ensemble",
]


MODEL_COLORS = {
    "Ridge": "#1f77b4",
    "Ridge_Ext": "#aec7e8",
    "Bayes_Ridge": "#ff7f0e",
    "ElasticNet": "#ffbb78",
    "Huber": "#2ca02c",
    "Ridge_Shock": "#98df8a",
    "Ridge_Shock_Roll24": "#66c2a5",
    "Ridge_ProdProxy": "#1b9e77",
    "Ridge_AsymERPT": "#7570b3",
    "Ridge_Macro": "#2ecc71",
    "Rolling_Ridge": "#e74c3c",  # Experimental model (red)
    "NGBoost": "#d62728",
    "NGBoost_Shock": "#ff9896",
    "BVAR": "#9467bd",
    "SARIMA": "#c5b0d5",
    "LightGBM": "#8c564b",
    "Prophet": "#c49c94",
    "ETS": "#e377c2",
    "EBM": "#f7b6d2",
    "CatBoost": "#7f7f7f",
    "Subcomp": "#c7c7c7",
    "Subcomp_Multi": "#bcbd22",
    "Micro": "#17becf",
    "Micro_SM": "#1f9e89",
    "Ensemble": "#000000",
    "Actual": "#000000",
    "Факт": "#000000",
}


MONTH_NAMES_RU = [
    "Январь",
    "Февраль",
    "Март",
    "Апрель",
    "Май",
    "Июнь",
    "Июль",
    "Август",
    "Сентябрь",
    "Октябрь",
    "Ноябрь",
    "Декабрь",
]


__all__ = [
    "ALL_MODELS",
    "MODEL_COLORS",
    "MONTH_NAMES_RU",
]
