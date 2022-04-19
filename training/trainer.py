import torch


class Trainer:
    """Implementation of a model training loop"""

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        lr_scheduler,
        criterion,
        num_epochs
    ):
        """
        Args:
            model (object): A pytorch model
            dataloader (object): The training
                loader on which the model should be trained
            optimizer (object): The chosen optimizer
            lr_scheduler (object): The chosen learning rate
                scheduler
            criterion (object): The chosen loss function
            num_epochs (int): The number of epochs for which the
                network should be trained
    
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.lr_scheduler_cls = lr_scheduler

    def _reset(self):
        """Helper function to reset values prior to training"""
        self.lr_scheduler = self.lr_scheduler_cls(
            self.optimizer,
            self.num_epochs
        )

    def _cost_function(self, **kwargs):
        """A general cost function"""
        forward_args = self.criterion.forward.__code__.co_varnames
        for key in list(kwargs):
            if key not in forward_args:
                del kwargs[key]
        return self.criterion(**kwargs)

    def train(self):
        """Run the training loop"""
        self._reset()
        self.model.train()
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.dataloader):
                images = images.to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)

                outputs = self.model(images)
                loss = self._cost_function(
                    input=outputs,
                    target=labels,
                    images=images
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
            
            print(f'Epoch {epoch}, loss={loss:2f}')
            