import numpy as np
import pymc as pm            # PyMC v5+ core API
import arviz as az           # InferenceData container and I/O

def fit_bayesian_logreg(
    X: np.ndarray,
    y: np.ndarray,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    random_seed: int | None = None
) -> tuple[pm.Model, az.InferenceData]:
    """
    Fit a Bayesian logistic regression model using PyMC v5 with mutable Data containers.
    Returns the compiled model and its InferenceData.
    """
    with pm.Model() as model:
        # ▶️ Register data as mutable containers for easy swapping later
        X_shared = pm.Data("X_shared", X)                      # :contentReference[oaicite:9]{index=9}
        y_shared = pm.Data("y_shared", y)

        # ▶️ Prior definitions
        alpha = pm.Normal("alpha", 0, 5)
        betas = pm.Normal("betas", 0, 5, shape=X.shape[1])

        # ▶️ Likelihood
        logit_p = alpha + pm.math.dot(X_shared, betas)
        pm.Bernoulli("obs", logit_p=logit_p, observed=y_shared)

        # ▶️ Sample; returns an ArviZ InferenceData container by default
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            return_inferencedata=True
        )                                                     # :contentReference[oaicite:10]{index=10}
    return model, idata

def save_pymc_inference(
    idata: az.InferenceData,
    path: str
) -> str:
    """
    Persist an ArviZ InferenceData object to a NetCDF file.
    Returns the filepath written.
    """
    # ▶️ Write to NetCDF; self-describing, versioned archive
    filepath = idata.to_netcdf(path)                        # :contentReference[oaicite:11]{index=11}
    return filepath

def load_pymc_inference(
    path: str
) -> az.InferenceData:
    """
    Reload a NetCDF file into an ArviZ InferenceData object.
    """
    idata = az.InferenceData.from_netcdf(path)               # :contentReference[oaicite:12]{index=12}
    return idata

def predict_bayesian_logreg(
    model: pm.Model,
    idata: az.InferenceData,
    X_new: np.ndarray,
    var_name: str = "obs",
    random_seed: int | None = None
) -> np.ndarray:
    """
    Perform out-of-sample posterior predictive classification.
    Swaps in X_new via pm.set_data, samples predictive draws, and thresholds at 0.5.
    """
    # ▶️ Swap in new data
    pm.set_data({"X_shared": X_new}, model=model)            # :contentReference[oaicite:13]{index=13}

    # ▶️ Generate posterior predictive draws as an InferenceData
    ppc = pm.sample_posterior_predictive(
        idata,
        model=model,
        var_names=[var_name],
        random_seed=random_seed,
        progressbar=False,
        return_inferencedata=True
    )                                                        # :contentReference[oaicite:14]{index=14}

    # ▶️ Extract samples and compute mean probabilities
    samples = ppc.posterior_predictive[var_name].values      # shape (chain, draw, obs)
    mean_preds = samples.mean(axis=(0, 1))
    return (mean_preds > 0.5).astype(int)


if __name__ == "__main__":
    # from src.nfl_kicker_analysis.utils.model_utils import (
    #     fit_bayesian_logreg,
    #     save_pymc_inference,
    #     load_pymc_inference,
    #     predict_bayesian_logreg
    # )
    import numpy as np

    # ▶️ Simulate toy data
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = rng.integers(0, 2, size=100)

    # ▶️ Fit model
    model, idata = fit_bayesian_logreg(X, y, draws=200, tune=100, chains=1)
    # ▶️ Save / reload inference
    nc_path = save_pymc_inference(idata, "test_bayes_logreg.nc")
    idata2 = load_pymc_inference(nc_path)
    # ▶️ Predict on training data
    preds = predict_bayesian_logreg(model, idata2, X, random_seed=1)

    # ▶️ Simple consistency check
    assert preds.shape == y.shape
    print("✅ PyMC Bayesian logistic regression end-to-end smoke test passed!")
