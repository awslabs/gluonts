class EarlyStopping:
    def __init__(self, max_steps_no_improvement=10):
        self.n_steps_no_improvement = 0
        self.max_steps_no_improvement = max_steps_no_improvement
        self.prev_loss = None

    def __call__(self, loss):
        if self.is_best(loss):
            self.n_steps_no_improvement = 0
            self.prev_loss = loss
        else:
            self.n_steps_no_improvement += 1
        stop = self.n_steps_no_improvement >= self.max_steps_no_improvement
        return stop

    def is_best(self, loss):
        return self.prev_loss is None or loss <= self.prev_loss
