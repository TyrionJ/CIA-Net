from .cia_net import CIANet

__all__ = ['CIANet', 'support_networks']

support_networks = list(__all__)
support_networks.remove('support_networks')
