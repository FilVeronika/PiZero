def load_data():
    df = pd.read_csv('/Users/veronika/Downloads/clasdb_pi_0_p.txt', delimiter='\t', header=None)
    df.columns = ['Ebeam', 'W', 'Q2', 'cos_theta', 'phi', 'dsigma_dOmega', 'error', 'id']
    df['phi'] = df.phi.apply(lambda x: math.radians(x))
    df['cos_phi'] = df['phi'].apply(lambda x: math.cos(x))
    df['sin_phi'] = df['phi'].apply(lambda x: math.sin(x))
    df['Ebeam'] = df['Ebeam'].round(decimals=2)
    df = df.replace({"Ebeam": {2.45: 2.44, 1.65: 1.64}})
    df = df.drop(df[df['dsigma_dOmega'] == 0].index)
    df = df.drop('id', axis=1)
    df = df.reset_index(drop=True)
    df = df.iloc[df[['Ebeam', 'W', 'Q2', 'cos_theta', 'phi']].drop_duplicates().index]
    df = df.reset_index(drop=True)
    return df
