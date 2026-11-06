import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

print("Завдання 1")
n = 30
dates = pd.date_range("2026-10-01", periods=n)
users = np.random.randint(200, 600, n)
sessions = (users * np.random.uniform(1.2, 2.0, n)).astype(int)
revenue = (sessions * np.random.uniform(0.8, 1.5, n)).round(2)
df1 = pd.DataFrame({"date": dates, "users": users, "sessions": sessions, "revenue": revenue})
print(df1.head())
corr = df1[["users", "sessions", "revenue"]].corr()
print("Кореляційна матриця:")
print(corr)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].scatter(df1["users"], df1["sessions"], color="blue")
axs[0].set_xlabel("users")
axs[0].set_ylabel("sessions")
axs[0].set_title("users - sessions")
axs[1].scatter(df1["users"], df1["revenue"], color="green")
axs[1].set_xlabel("users")
axs[1].set_ylabel("revenue")
axs[1].set_title("users - revenue")
axs[2].scatter(df1["sessions"], df1["revenue"], color="red")
axs[2].set_xlabel("sessions")
axs[2].set_ylabel("revenue")
axs[2].set_title("sessions - revenue")
for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df1["date"], df1["revenue"], marker="o", color="purple")
plt.xlabel("Дата")
plt.ylabel("Виручка")
plt.title("Виручка за датами")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nЗавдання 2")
size = 200
group = np.repeat(["A", "B"], size)
converted = np.concatenate([
    np.random.binomial(1, 0.10, size),
    np.random.binomial(1, 0.14, size)
])
df2 = pd.DataFrame({"group": group, "converted": converted})
stats = df2.groupby("group")["converted"].agg(["sum", "count", "mean"])
stats.columns = ["conversions", "total", "rate"]
print(stats)

rate_a = stats.loc["A", "rate"]
rate_b = stats.loc["B", "rate"]
abs_diff = rate_b - rate_a
rel_change = abs_diff / rate_a * 100
print(f"Конверсія A: {rate_a:.4f}")
print(f"Конверсія B: {rate_b:.4f}")
print(f"Абсолютна різниця: {abs_diff:.4f}")
print(f"Відносна зміна: {rel_change:.2f}%")

z = 1.96
errors = []
for g in ["A", "B"]:
    p = stats.loc[g, "rate"]
    cnt = stats.loc[g, "total"]
    se = np.sqrt(p * (1 - p) / cnt)
    low = p - z * se
    high = p + z * se
    errors.append(z * se)
    print(f"95% ДІ для групи {g}: [{low:.4f}, {high:.4f}]")

plt.figure(figsize=(6, 5))
plt.bar(["A", "B"], [rate_a, rate_b], yerr=errors, capsize=8, color=["skyblue", "lightgreen"], edgecolor="black")
plt.xlabel("Група")
plt.ylabel("Конверсія")
plt.title("Конверсія груп із 95% довірчими інтервалами")
plt.show()

print("\nЗавдання 3")
population = np.random.exponential(scale=2.0, size=50000)
print("Середнє генеральної сукупності:", population.mean())
print("Стандартне відхилення генеральної сукупності:", population.std())

sizes = [5, 30, 100]
fig, axs = plt.subplots(1, len(sizes), figsize=(15, 4))
for ax, k in zip(axs, sizes):
    means = []
    for _ in range(2000):
        sample = np.random.choice(population, k)
        means.append(sample.mean())
    means = pd.Series(means)
    print(f"n = {k}: середнє = {means.mean():.4f}, стандартне відхилення = {means.std():.4f}")
    ax.hist(means, bins=30, color="orange", edgecolor="black")
    ax.set_xlabel("Вибіркове середнє")
    ax.set_ylabel("Частота")
    ax.set_title(f"n = {k}")
fig.suptitle("Розподіл вибіркових середніх")
plt.tight_layout()
plt.show()

print("\nЗавдання 4")
days = pd.date_range("2026-08-01", periods=90)
trend = np.linspace(100, 160, 90)
noise = np.random.normal(0, 12, 90)
sales = (trend + noise).round(2)
df4 = pd.DataFrame({"date": days, "sales": sales})
window = 7
df4["rolling_mean"] = df4["sales"].rolling(window).mean()
df4["rolling_std"] = df4["sales"].rolling(window).std()
print(df4.head(10))

plt.figure(figsize=(11, 5))
plt.plot(df4["date"], df4["sales"], label="Продажі", color="lightblue")
plt.plot(df4["date"], df4["rolling_mean"], label=f"Ковзне середнє ({window} днів)", color="red")
plt.xlabel("Дата")
plt.ylabel("Продажі")
plt.title("Продажі та ковзне середнє")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(11, 5))
plt.plot(df4["date"], df4["rolling_std"], color="green")
plt.xlabel("Дата")
plt.ylabel("Стандартне відхилення")
plt.title(f"Ковзне стандартне відхилення ({window} днів)")
plt.grid(True)
plt.show()
