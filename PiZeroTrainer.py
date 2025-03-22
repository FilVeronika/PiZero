class PiZeroTrainer:
    def __init__(self, df, hyperparams):
        self.df = df
        self.hyperparams = hyperparams
        self.models = {}
        self.trainers = {}
        self.loss_histories = {}

    def split_data_by_energy(self):
        energy_groups = self.df['Ebeam'].unique()
        datasets = {}
        for energy in energy_groups:
            datasets[energy] = self.df[self.df['Ebeam'] == energy]
        return datasets

    def train_models(self):
        energy_datasets = self.split_data_by_energy()
        for energy, dataset in energy_datasets.items():
            train_df, val_df = train_test_split(dataset, test_size=0.2, random_state=42)
            train_dataset = PiZeroDataset(train_df)
            val_dataset = PiZeroDataset(val_df)
            train_loader = DataLoader(train_dataset, batch_size=self.hyperparams["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.hyperparams["batch_size"], shuffle=False)

            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath='checkpoints/',
                filename=f'best-model-{{epoch:02d}}-{{val_loss:.2f}}',
                save_top_k=1,
                mode='min',
            )
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=self.hyperparams["early_stopping_patience"],
                verbose=True,
                mode='min'
            )

            trainer = pl.Trainer(
                max_epochs=self.hyperparams["max_epochs"],
                callbacks=[checkpoint_callback, early_stop_callback],
                log_every_n_steps=10
            )

            model = PiZeroModel(self.hyperparams)
            trainer.fit(model, train_loader, val_loader)

            self.models[energy] = model
            self.loss_histories[energy] = {
                'train_losses': model.train_losses,
                'val_losses': model.val_losses
            }
    def plot_training_curves(self):
        for energy, losses in self.loss_histories.items():
            print(f"Model for Ebeam = {energy}")
            
            train_losses = losses['train_losses']
            val_losses = losses['val_losses']
            
            if not train_losses or not val_losses:
                print(f"No training/validation data for Ebeam = {energy}")
                continue
            
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (RMSE)')
            plt.title(f'Training and Validation Loss for Ebeam = {energy}')
            plt.legend()
            plt.show()
