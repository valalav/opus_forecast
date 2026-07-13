from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

router = APIRouter(prefix="/explain", tags=["SHAP Explanations"])


class ExplainRequest(BaseModel):
    model_name: str = "random_forest"
    features: List[Dict[str, float]]
    target_index: int = 0


class FeatureImportance(BaseModel):
    feature_name: str
    importance: float
    contribution: float


class ExplainResponse(BaseModel):
    model_name: str
    feature_importance: List[Dict[str, Any]]
    shap_values: List[List[float]]
    base_value: float


_global_model = None
_feature_names = None


def _get_demo_model():
    global _global_model, _feature_names
    if _global_model is None:
        np.random.seed(42)
        X_train = np.random.rand(100, 5)
        y_train = (
            X_train[:, 0] * 0.3
            + X_train[:, 1] * 0.2
            + X_train[:, 2] * 0.4
            + X_train[:, 3] * 0.1
            + np.random.randn(100) * 0.1
        )
        _global_model = RandomForestRegressor(n_estimators=50, random_state=42)
        _global_model.fit(X_train, y_train)
        _feature_names = [f"feature_{i}" for i in range(5)]
    return _global_model, _feature_names


@router.get("")
@router.get("/")
async def explain_root():
    return {"message": "SHAP Explanation API", "endpoints": ["/explain/", "/explain"]}


@router.post("/", response_model=ExplainResponse)
async def explain_prediction(request: ExplainRequest):
    try:
        model, feature_names = _get_demo_model()

        if not request.features:
            raise HTTPException(status_code=400, detail="No features provided")

        feature_values = []
        if isinstance(request.features[0], dict):
            feature_values = [list(f.values())[0] for f in request.features]
        else:
            feature_values = [f for f in request.features]

        if len(feature_values) < 5:
            feature_values = feature_values + [0.0] * (5 - len(feature_values))
        else:
            feature_values = feature_values[:5]

        X = np.array(feature_values).reshape(1, -1)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[0])

        feature_importance = []
        for i, name in enumerate(feature_names):
            imp = abs(float(shap_values[0][i]))
            contrib = float(shap_values[0][i])
            feature_importance.append(
                {"feature_name": name, "importance": imp, "contribution": contrib}
            )

        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        return ExplainResponse(
            model_name=request.model_name,
            feature_importance=feature_importance,
            shap_values=shap_values.tolist(),
            base_value=float(base_value),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error computing SHAP values: {str(e)}"
        )


@router.get("/features")
async def get_features():
    model, feature_names = _get_demo_model()
    return {
        "model_name": "random_forest",
        "features": feature_names,
        "feature_count": len(feature_names),
    }


@router.get("/importance")
async def get_global_importance():
    try:
        model, feature_names = _get_demo_model()

        explainer = shap.TreeExplainer(model)

        X_background = np.random.rand(50, 5)
        shap_values_bg = explainer.shap_values(X_background)

        if isinstance(shap_values_bg, list):
            shap_values_bg = shap_values_bg[0]

        mean_abs_shap = np.mean(np.abs(shap_values_bg), axis=0)

        importance = []
        for i, name in enumerate(feature_names):
            importance.append(
                {"feature_name": name, "mean_importance": float(mean_abs_shap[i])}
            )

        importance.sort(key=lambda x: x["mean_importance"], reverse=True)

        return {"model_name": "random_forest", "global_feature_importance": importance}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error computing global importance: {str(e)}"
        )
