class Config:

    def __init__(self):
        self.grid = 84
        self.num_emb_state = 64 + 2 + 1
        self.soft_coefficient = 1
        self.num_state = 1 + self.grid * self.grid * 5 + 2

config = Config()