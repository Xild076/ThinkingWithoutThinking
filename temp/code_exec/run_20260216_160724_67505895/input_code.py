try:
    import pandas as pd
    df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
    df_clean = df.dropna()
    df2 = pd.DataFrame({'A': [1,2], 'C': [10,20]})
    df_merged = pd.merge(df_clean, df2, on='A', how='left')
    df_merged['C'] = df_merged['C'].fillna(0)
    result = len(df_merged)
except ImportError:
    data = [{'A':1, 'B':4}, {'A':None, 'B':5}, {'A':3, 'B':None}]
    df_clean = [d for d in data if d.get('A') is not None]
    merge_dict = {1: {'C':10}, 2: {'C':20}}
    df_merged = []
    for row in df_clean:
        a = row['A']
        c = merge_dict.get(a, {})
        merged = {**row, 'C': c.get('C', 0)}
        df_merged.append(merged)
    result = len(df_merged)
print(result)