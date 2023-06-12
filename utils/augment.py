import torch


class Noise(object):
    def __init__(self, am=0.5) -> None:
        super().__init__()
        self.am = am

    def __call__(self, input):
        rand_t = torch.rand(input.size()) - 0.5
        to_add_t = rand_t * 2 * self.am
        input = input + to_add_t
        return input


class Transform(object):
    def __init__(self, type, gs_am=0.5):
        assert type is not None
        self.type  = type
        self.gs_am = gs_am

    def __call__(self, input):
        if self.type == 'noise':
            transform = Noise(am=self.gs_am)(input)
        else:
            raise Exception('No such transform')

        return transform
