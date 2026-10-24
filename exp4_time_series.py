import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)

pd.set_option('display.width', 150)

df = pd.read_csv("Sub_Division_IMD_2017.csv")
print("Shape:", df.shape)
print("Subdivisions:", df['SUBDIVISION'].nunique(), " Years:", df['YEAR'].min(), "-", df['YEAR'].max())
print(df.head())
df.info()

df['date'] = pd.to_datetime(df['YEAR'].astype(str), format='%Y')
df = df.sort_values(['SUBDIVISION', 'date']).reset_index(drop=True)
print(df[['SUBDIVISION', 'YEAR', 'date']].head(3))

sensors = ['JF', 'MAM', 'JJAS', 'OND', 'ANNUAL']
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
fill_cols = months + sensors

print("Missing values before:", df[fill_cols].isnull().sum().sum())

df[fill_cols] = df.groupby('SUBDIVISION')[fill_cols].transform(lambda s: s.interpolate(limit_direction='both'))
print("Missing values after interpolation:", df[fill_cols].isnull().sum().sum())

full = pd.MultiIndex.from_product([df['SUBDIVISION'].unique(), range(1901, 2018)], names=['SUBDIVISION', 'YEAR'])
ts = df.set_index(['SUBDIVISION', 'YEAR'])[sensors].reindex(full).reset_index()
print("Years missing completely (left empty, not invented):", ts['ANNUAL'].isnull().sum())

goa = df[df['SUBDIVISION'] == 'Konkan & Goa'].set_index('date')

fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
for ax, col in zip(axes, sensors):
    ax.plot(goa.index, goa[col], linewidth=1)
    ax.set_ylabel(col + " (mm)")
axes[0].set_title("Konkan & Goa - rainfall measurements over time")
axes[-1].set_xlabel("Year")
plt.tight_layout()
plt.savefig("fig1_time_series.png", dpi=120)
plt.show()

variation = pd.DataFrame({
    'CV %': goa[sensors].std() / goa[sensors].mean() * 100,
    'avg yearly change %': goa[sensors].diff().abs().mean() / goa[sensors].mean() * 100,
})
print(variation.round(1))

goa['MA5'] = goa['ANNUAL'].rolling(window=5).mean()
goa['MA10'] = goa['ANNUAL'].rolling(window=10).mean()

plt.figure(figsize=(11, 4.5))
plt.plot(goa.index, goa['ANNUAL'], color='lightgray', label='ANNUAL')
plt.plot(goa.index, goa['MA5'], label='5-year moving average')
plt.plot(goa.index, goa['MA10'], label='10-year moving average', linewidth=2)
plt.ylabel("Rainfall (mm)")
plt.title("Konkan & Goa - ANNUAL rainfall with moving averages")
plt.legend()
plt.tight_layout()
plt.savefig("fig2_moving_average.png", dpi=120)
plt.show()

slope, intercept, r, p, se = stats.linregress(goa['YEAR'], goa['ANNUAL'])
print(f"Trend: {slope * 10:.1f} mm per decade, p-value = {p:.3f}")

goa['STD10'] = goa['ANNUAL'].rolling(window=10).std()

plt.figure(figsize=(11, 4))
plt.plot(goa.index, goa['STD10'], color='tab:orange')
plt.ylabel("Rolling std (mm)")
plt.title("Konkan & Goa - 10-year rolling standard deviation of ANNUAL rainfall")
plt.tight_layout()
plt.savefig("fig3_rolling_std.png", dpi=120)
plt.show()

goa['past_mean'] = goa['ANNUAL'].rolling(10).mean().shift(1)
goa['past_std'] = goa['ANNUAL'].rolling(10).std().shift(1)
goa['z'] = (goa['ANNUAL'] - goa['past_mean']) / goa['past_std']
goa['anomaly'] = goa['z'].abs() > 2

checked = goa['z'].notnull().sum()
n_anom = goa['anomaly'].sum()
print(f"Konkan & Goa: {n_anom} anomalies out of {checked} years checked = {n_anom / checked * 100:.1f}%")
print(goa.loc[goa['anomaly'], ['YEAR', 'ANNUAL', 'z']].round(2).to_string())

plt.figure(figsize=(11, 4.5))
plt.plot(goa.index, goa['ANNUAL'], label='ANNUAL')
plt.scatter(goa.index[goa['anomaly']], goa['ANNUAL'][goa['anomaly']], color='red', zorder=3, label='anomaly')
plt.ylabel("Rainfall (mm)")
plt.title("Konkan & Goa - anomalies in ANNUAL rainfall")
plt.legend()
plt.tight_layout()
plt.savefig("fig4_anomalies.png", dpi=120)
plt.show()

g = ts.groupby('SUBDIVISION')['ANNUAL']
ts['z'] = (ts['ANNUAL'] - g.transform(lambda s: s.rolling(10).mean().shift(1))) / \
          g.transform(lambda s: s.rolling(10).std().shift(1))
ts['anomaly'] = ts['z'].abs() > 2

checked_all = ts['z'].notnull().sum()
print(f"All subdivisions: {ts['anomaly'].sum()} anomalies out of {checked_all} = {ts['anomaly'].sum() / checked_all * 100:.1f}%")
print("  too wet (z > 2):", (ts['z'] > 2).sum(), "  too dry (z < -2):", (ts['z'] < -2).sum())

split_year = 1995

train_mean = ts[ts['YEAR'] <= split_year].groupby('SUBDIVISION')['ANNUAL'].mean()
ts['rel'] = ts['ANNUAL'] / ts['SUBDIVISION'].map(train_mean) * 100

g = ts.groupby('SUBDIVISION')['rel']
ts['rel_now'] = ts['rel']
ts['rel_prev'] = g.shift(1)
ts['ma5'] = g.transform(lambda s: s.rolling(5).mean())
ts['ma10'] = g.transform(lambda s: s.rolling(10).mean())
ts['std5'] = g.transform(lambda s: s.rolling(5).std())
ts['diff1'] = g.diff()
ts['trend'] = ts['ma5'] - ts['ma10']

for c in ['JF', 'MAM', 'JJAS', 'OND']:
    ts[c + '_share'] = ts[c] / ts['ANNUAL'] * 100

features = ['rel_now', 'rel_prev', 'ma5', 'ma10', 'std5', 'diff1', 'trend',
            'JF_share', 'MAM_share', 'JJAS_share', 'OND_share']
print(ts[['SUBDIVISION', 'YEAR'] + features].dropna().head(3).round(1).to_string())

ts['drought'] = np.where(ts['rel'].isnull(), np.nan, (ts['rel'] < 80).astype(float))
ts['target'] = ts.groupby('SUBDIVISION')['drought'].shift(-1)

data = ts.dropna(subset=features + ['target']).copy()
data['target'] = data['target'].astype(int)
print("Rows for modelling:", len(data))
print("Failure years (target = 1):", data['target'].sum(), f"({data['target'].mean() * 100:.1f}%)")

train = data[data['YEAR'] + 1 <= split_year]
test = data[data['YEAR'] + 1 > split_year]

X_train, y_train = train[features], train['target']
X_test, y_test = test[features], test['target']

print(f"Train: {len(train)} rows, target years up to {split_year}, failure rate {y_train.mean() * 100:.1f}%")
print(f"Test : {len(test)} rows, target years {split_year + 1}-2017, failure rate {y_test.mean() * 100:.1f}%")

model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=10,
                               class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
print("Model trained on", len(X_train), "rows with", len(features), "features")

def scores(name, y_true, y_pred, y_prob=None):
    row = {'model': name,
           'accuracy': accuracy_score(y_true, y_pred),
           'precision': precision_score(y_true, y_pred, zero_division=0),
           'recall': recall_score(y_true, y_pred, zero_division=0),
           'F1': f1_score(y_true, y_pred, zero_division=0)}
    row['ROC-AUC'] = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    return row

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

results = pd.DataFrame([scores('Random Forest', y_test, pred, prob),
                        scores('Always "no failure"', y_test, baseline.predict(X_test))])
print(results.round(3).to_string(index=False))

cm = confusion_matrix(y_test, pred)
print("\nConfusion matrix (rows = actual, columns = predicted)")
print(pd.DataFrame(cm, index=['actual normal', 'actual failure'], columns=['pred normal', 'pred failure']))

importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(importance.round(3).to_string())

plt.figure(figsize=(8, 4.5))
importance.sort_values().plot(kind='barh', color='tab:green')
plt.xlabel("Feature importance")
plt.title("Random Forest - which features matter for predicting failure")
plt.tight_layout()
plt.savefig("fig5_feature_importance.png", dpi=120)
plt.show()

check = data[data['z'].notnull()]
after_anomaly = check['z'].abs() > 2
rate_after_anomaly = check.loc[after_anomaly, 'target'].mean() * 100
rate_after_normal = check.loc[~after_anomaly, 'target'].mean() * 100
print(f"Failure next year after an anomalous year : {rate_after_anomaly:.1f}%  ({after_anomaly.sum()} cases)")
print(f"Failure next year after a normal year     : {rate_after_normal:.1f}%  ({(~after_anomaly).sum()} cases)")
