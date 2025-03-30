from ood_detectors.interface import OODDetector

def create_ood_detector(name: str) -> OODDetector:
    if name == "mahalanobis":
        from ood_detectors.mahalanobis import MahalanobisOODDetector
        return MahalanobisOODDetector()
    else:
        raise NotImplementedError()