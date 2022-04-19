import torch.nn.functional as F


class KDLoss:
    """Implementation of knowledge distillation loss
    
    Knowledge distillation loss is a combination of traditional
    cross entropy loss (on hard labels) and kullback-leibler
    divergence loss between the outputs of the student and 
    teacher models. A student model is a single MLP model, which
    is learning to match the output distribution of a deep ensemble.

    """

    def __init__(self, deep_ensemble, alpha, temp):
        """
        Args:
        deep_ensemble (DeepEnsemble): The deep ensemble model which will
            act as the teacher
        alpha (float): The proportion of KL divergence loss in the final
            loss. 1-alpha is the proportion of CrossEntropy loss.
        temp (float): The temperature of the softmax function. Higher
            temperatures lead to softer probability distributions. In 
            other words, larger dispersion of probability mass.
        
        """
        self.deep_ensemble = deep_ensemble
        self.alpha = alpha
        self.temp = temp

    def calc_dist_p(self, student_outputs):
        """ Calculate the log of the ditribution of the student outputs

        The torch.nn.functional.kl_div function expects distriubtion
        p in log format.

        Args:
            student_outputs (torch.tensor): The outputs of the student
                MLP model
        
        Returns:
            p (torch.tensor): The log probabilities of the student
                outputs
        
        """
        return F.log_softmax(student_outputs / self.temp, dim=1)

    def calc_dist_q(self, images):
        """ Calculate the distribution of teacher outputs

        Args:
            images (torch.tensor): A batch of images on which the
                teacher model should be evaluated
        
        Returns:
            q (torch.tensor): The probabilities of the teacher
                outputs
        
        """
        teacher_outputs = self.deep_ensemble(images)
        return F.softmax(teacher_outputs / self.temp, dim=1)

    def calc_kl_divergence(self, p, q):
        """KL-divergence between distributions p and q

        Note that KL-divergence is scaled by temp ** 2. This is
        due to the fact that the gradients of the soft targets
        scale at 1 / (temp ** 2). Without it the relative contributions
        of KL_loss and CE_loss would change over time.

        Args:
            p (torch.tensor): The log probabilities of the student
                outputs
            q (torch.tensor): The probabilities of the teacher outputs
        
        Returns:
            KL_loss (torch.tensor): KL-divergence loss
        """
        return (F.kl_div(p, q, size_average=False) * (self.temp ** 2)) / p.shape[0]
    
    def calc_cross_entropy(self, student_outputs, labels):
        """Traditional cross entropy loss
        
        Args:
            student_outputs (torch.tensor): The outputs of the student
                MLP model
            labels (torch.tensor): The ground truth class labels

        Returns:
            CE_loss (torch.tensor): Cross Entropy loss
        
        """
        return F.cross_entropy(student_outputs, labels)
    
    def forward(self, input, target, images):
        """Calculate knowledge distilattion loss

        Args:
            input (torch.tensor): The outputs of the student
                MLP model
            target (torch.tensor): The ground truth class labels
            images (torch.tensor): A batch of images on which the
                teacher model should be evaluated

        Returns:
            KD_loss (torch.tensor): Knowledge distillation loss
        
        """
        p = self.calc_dist_p(input)
        q = self.calc_dist_q(images)

        KL_loss = self.calc_kl_divergence(p, q)
        CE_loss = self.calc_cross_entropy(input, target)

        return self.alpha * KL_loss + (1 - self.alpha) * CE_loss

    def __call__(self, input, target, images):
        """ Calls forward function

        Args:
            Args:
            input (torch.tensor): The outputs of the student
                MLP model
            target (torch.tensor): The ground truth class labels
            images (torch.tensor): A batch of images on which the
                teacher model should be evaluated

        Returns:
            KD_loss (torch.tensor): Knowledge distillation loss

        """
        return self.forward(
            input=input,
            target=target,
            images=images
        )

