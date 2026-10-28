import time
import pandas as pd
from matbench.bench import MatbenchBenchmark
from matminer.featurizers.composition import ElementProperty

mb = MatbenchBenchmark(autoload=False, subset=["matbench_log_gvrh"])
t = mb.matbench_log_gvrh
t.load()
df = t.df.head(200)
ep = ElementProperty.from_preset("magpie")
t0 = time.time()
X = ep.featurize_many([s.composition for s in df["structure"]], ignore_errors=True, n_jobs=1)
print("time200", round(time.time() - t0, 2))
X = pd.DataFrame(X, columns=ep.feature_labels())
print(X.shape, "nans", int(X.isna().sum().sum()))
print(X.columns[:6].tolist())
