import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def lossn(pred,opt,target):
    loss_fn = focal_loss(alpha=0.25, gamma=2, class_num=opt.label_dim)
    loss=loss_fn(pred,target)
    return(loss)
#class FocalLoss(nn.Module):
#    r"""
#        This criterion is a implemenation of Focal Loss, which is proposed in 
#        Focal Loss for Dense Object Detection.
#
#            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#
#        The losses are averaged across observations for each minibatch.
#
#        Args:
#            alpha(1D Tensor, Variable) : the scalar factor for this criterion
#            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
#                                   putting more focus on hard, misclassiﬁed examples
#            size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                However, if the field size_average is set to False, the losses are
#                                instead summed for each minibatch.
#
#
#    """
#    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
#        super(FocalLoss, self).__init__()
#        if alpha is None:
#            self.alpha = Variable(torch.ones(class_num, 1))
#        else:
#            if isinstance(alpha, Variable):
#                self.alpha = alpha
#            else:
#                self.alpha = Variable(alpha)
#        self.gamma = gamma
#        self.class_num = class_num
#        self.size_average = size_average
#
#    def forward(self, inputs, targets):
#        N = inputs.size(0)
#        C = inputs.size(1)
#        P = F.softmax(inputs)
#
#        class_mask = inputs.data.new(N, C).fill_(0)
#        class_mask = Variable(class_mask)
#        ids = targets.view(-1, 1)
#        class_mask.scatter_(1, ids.data, 1.)
#        #print(class_mask)
#
#
#        if inputs.is_cuda and not self.alpha.is_cuda:
#            self.alpha = self.alpha.cuda()
#        alpha = self.alpha[ids.data.view(-1)]
#
#        probs = (P*class_mask).sum(1).view(-1,1)
#
#        log_p = probs.log()
#        #print('probs size= {}'.format(probs.size()))
#        #print(probs)
#
#        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
#        #print('-----bacth_loss------')
#        #print(batch_loss)
#
#
#        if self.size_average:
#            loss = batch_loss.mean()
#        else:
#            loss = batch_loss.sum()
#        return loss
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2,class_num = 3, size_average=True):
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  
            self.alpha = torch.zeros(class_num)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) 
        preds_logsoft = torch.log(preds_softmax)
        
        #focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
