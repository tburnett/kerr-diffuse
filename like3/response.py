"""
Manage the instrument rsponse for non-diffuse sources
"""

class Response:
    def __init__(self, source, band, roi=None, **kwargs):
        """
        Given a source and a band, set values for set of pixels
        """    
        self.source = source
        self.band = band   
        raise NotImplementedError(f'Called with source {source.name}')


class PointResponse(Response):
    pass
    
class ExtendedResponse(Response):
    pass
