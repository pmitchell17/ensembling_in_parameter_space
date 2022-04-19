import torch

class Evaluator:
    """Evaluate the accuracy of a given model on a test set"""

    def __init__(self, test_loader):
        """
        Args:
            test_loader (torch.utils.data.Dataloader): A dataloader
                of the test set
        """
        self.test_loader = test_loader
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
    
    def evaluate(self, model):
        """Calculate the accuracy of a model
        
        Args:
            model (nn.Module): A pytorch model which should be
                evaluated

        Returns:
            accuracy (float): The accuracy of the model on the
                test dataset

        """
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

