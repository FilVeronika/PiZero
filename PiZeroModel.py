class PiZeroModel(pl.LightningModule):
    def __init__(self, hyperparams):
        super(PiZeroModel, self).__init__()
        self.hyperparams = hyperparams
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(hyperparams["input_size"], hyperparams["hidden_layers"][0]))
        for i in range(1, len(hyperparams["hidden_layers"])):
            self.layers.append(nn.Linear(hyperparams["hidden_layers"][i-1], hyperparams["hidden_layers"][i]))
        self.layers.append(nn.Linear(hyperparams["hidden_layers"][-1], hyperparams["output_size"]))
        self.learning_rate = hyperparams["learning_rate"]
        self.train_losses = []
        self.val_losses = []
        self.epoch_train_loss = []
        self.epoch_val_loss = []

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def training_step(self, batch, batch_idx):
        features = batch['features']
        targets = batch['target']
        weights = batch['weights']
        predictions = self(features)
        loss = torch.sqrt(torch.mean(weights * (predictions - targets) ** 2))
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.epoch_train_loss.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        features = batch['features']
        targets = batch['target']
        predictions = self(features)
        loss = torch.sqrt(torch.mean((predictions - targets) ** 2))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.epoch_val_loss.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        avg_train_loss = sum(self.epoch_train_loss) / len(self.epoch_train_loss)
        self.train_losses.append(avg_train_loss)
        self.epoch_train_loss.clear()

    def on_validation_epoch_end(self):
        avg_val_loss = sum(self.epoch_val_loss) / len(self.epoch_val_loss)
        self.val_losses.append(avg_val_loss)
        self.epoch_val_loss.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.hyperparams["reduce_lr_patience"], verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
