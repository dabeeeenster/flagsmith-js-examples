import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Setup Parameters
np.random.seed(42)
days = 14
n_per_day = 1000
baseline_rate = 0.10  # 10% Baseline
variant_rate = 0.12  # 12% Actual Performance (2% lift)

# 2. Generate Simulated Data
timestamps = pd.date_range(start="2023-01-01", periods=days, freq="D")
data = []

for date in timestamps:
    conversions = np.random.binomial(n=n_per_day, p=variant_rate)
    data.append({"timestamp": date, "conversions": conversions, "visitors": n_per_day})

df = pd.DataFrame(data)

# 3. Calculate Cumulative Metrics
df["cum_conversions"] = df["conversions"].cumsum()
df["cum_visitors"] = df["visitors"].cumsum()
df["cum_rate"] = df["cum_conversions"] / df["cum_visitors"]

# 4. Calculate 95% Confidence Interval (Z=1.96)
# Standard Error formula: sqrt(p * (1-p) / n)
df["std_err"] = np.sqrt(df["cum_rate"] * (1 - df["cum_rate"]) / df["cum_visitors"])
df["ci_upper"] = df["cum_rate"] + (1.96 * df["std_err"])
df["ci_lower"] = df["cum_rate"] - (1.96 * df["std_err"])

# 5. Log Results
print(f"A/B Test Simulation: {days} days, {n_per_day} visitors/day")
print(f"Baseline rate: {baseline_rate:.1%}, Variant rate: {variant_rate:.1%}")
print(f"Expected lift: {(variant_rate - baseline_rate) / baseline_rate:.1%}")
print()
print(df[["timestamp", "conversions", "visitors", "cum_rate", "ci_lower", "ci_upper"]].to_string(
    index=False,
    formatters={
        "timestamp": lambda x: x.strftime("%Y-%m-%d"),
        "cum_rate": "{:.4f}".format,
        "ci_lower": "{:.4f}".format,
        "ci_upper": "{:.4f}".format,
    },
))
print()
print(f"Final conversion rate: {df['cum_rate'].iloc[-1]:.4f}")
print(f"Final 95% CI: [{df['ci_lower'].iloc[-1]:.4f}, {df['ci_upper'].iloc[-1]:.4f}]")
print(f"Baseline ({baseline_rate:.2f}) {'outside' if df['ci_lower'].iloc[-1] > baseline_rate else 'inside'} CI -> "
      f"{'Statistically significant' if df['ci_lower'].iloc[-1] > baseline_rate else 'Not yet significant'}")

# 6. Plot Results
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["timestamp"], df["cum_rate"], color="#2563eb", linewidth=2, label="Cumulative Conversion Rate")
ax.fill_between(df["timestamp"], df["ci_lower"], df["ci_upper"], color="#2563eb", alpha=0.15, label="95% Confidence Interval")
ax.axhline(y=baseline_rate, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline_rate:.0%})")
ax.axhline(y=variant_rate, color="#16a34a", linestyle=":", linewidth=1.5, label=f"True Variant Rate ({variant_rate:.0%})")

ax.set_xlabel("Date")
ax.set_ylabel("Conversion Rate")
ax.set_title("A/B Test: Cumulative Conversion Rate with 95% CI")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()

output_path = "ab_test_results.png"
fig.savefig(output_path, dpi=150)
print(f"\nChart saved to {output_path}")
