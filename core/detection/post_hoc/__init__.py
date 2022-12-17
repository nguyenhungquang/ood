from .msp_energy import EnergyBasedDetector, MSPDetector
from .gan_d import GANDDetector
from .pMNL import pNMLDetector
from .local_ensemble import LocalEnsembleDetector
from .odin import ODINDetector
from .mahalanobis import MahalanobisDetector

def get_detector_from_name(name):
    d = {
        'msp': MSPDetector,
        'energy': EnergyBasedDetector,
        'odin': ODINDetector,
        'mahalanobis': MahalanobisDetector,
        'M': MahalanobisDetector,
        'pNML': pNMLDetector,
        'local_ensemble': LocalEnsembleDetector
    }
    return d[name]