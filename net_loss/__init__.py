from .cianet_loss import CIANetLoss

__all__ = ['CIANetLoss', 'support_loss_fns']

support_loss_fns = list(__all__)
support_loss_fns.remove('support_loss_fns')
