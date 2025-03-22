class PiZeroDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.features = dataframe[['Ebeam', 'W', 'Q2', 'cos_theta', 'cos_phi', 'sin_phi']].values
        self.targets = dataframe['dsigma_dOmega'].values
        self.weights = 1 / dataframe['error'].values
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32),
            'weights': torch.tensor(self.weights[idx], dtype=torch.float32)
        }
